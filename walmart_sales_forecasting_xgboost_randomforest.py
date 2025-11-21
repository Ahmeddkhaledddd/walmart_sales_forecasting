

# === Common Imports ===
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb


def custom_objective(preds, dtrain):
    labels = dtrain.get_label()
    peak_threshold = np.percentile(labels, 95)
    grad = np.where(labels > peak_threshold, 2 * (preds - labels), preds - labels)
    hess = np.where(labels > peak_threshold, 2.0, 1.0)
    return grad, hess


base_path = "/content/drive/MyDrive/"
stores = pd.read_csv(base_path + "stores.csv")
features = pd.read_csv(base_path + "features.csv")
train = pd.read_csv(base_path + "train.csv")
test = pd.read_csv(base_path + "test.csv")
train_merged = pd.merge(train, features, on=["Store", "Date", "IsHoliday"], how="left")
train_merged = pd.merge(train_merged, stores, on="Store", how="left")
test_merged = pd.merge(test, features, on=["Store", "Date", "IsHoliday"], how="left")
test_merged = pd.merge(test_merged, stores, on="Store", how="left")
train_merged['Date'] = pd.to_datetime(train_merged['Date'])
test_merged['Date'] = pd.to_datetime(test_merged['Date'])
for df in [train_merged, test_merged]:
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Day'] = df['Date'].dt.day
    df['IsHoliday'] = df['IsHoliday'].astype(int)
lag_features = [1, 2, 3, 7, 14, 21]
for lag in lag_features:
    train_merged[f'Lag_{lag}'] = train_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)
for lag in lag_features:
    test_merged[f'Lag_{lag}'] = np.nan

train_merged.fillna(train_merged.median(numeric_only=True), inplace=True)
test_merged.fillna(train_merged.median(numeric_only=True), inplace=True)
train_merged = pd.get_dummies(train_merged, columns=["Type"], prefix="Type")
test_merged = pd.get_dummies(test_merged, columns=["Type"], prefix="Type")
missing_cols = set(train_merged.columns) - set(test_merged.columns)
for col in missing_cols:
    if col.startswith("Type_"):
        test_merged[col] = 0
test_merged = test_merged[train_merged.columns.drop("Weekly_Sales", errors="ignore")]
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
train_merged[markdown_cols] = train_merged[markdown_cols].fillna(0)
test_merged[markdown_cols] = test_merged[markdown_cols].fillna(0)
train_merged['Total_MarkDown'] = train_merged[markdown_cols].sum(axis=1)
test_merged['Total_MarkDown'] = test_merged[markdown_cols].sum(axis=1)
train_merged[['CPI', 'Unemployment']] = train_merged[['CPI', 'Unemployment']].ffill()
test_merged[['CPI', 'Unemployment']] = test_merged[['CPI', 'Unemployment']].ffill()

train_merged['Holiday_Dept'] = train_merged['IsHoliday'] * train_merged['Dept']
test_merged['Holiday_Dept'] = test_merged['IsHoliday'] * test_merged['Dept']
train_merged['Weekly_Sales'] = np.clip(train_merged['Weekly_Sales'], 0, train_merged['Weekly_Sales'].quantile(0.99))
peak_threshold = train_merged['Weekly_Sales'].quantile(0.95)
train_merged['Is_Peak_Week'] = (train_merged['Weekly_Sales'] > peak_threshold).astype(int)
test_merged['Is_Peak_Week'] = 0
train_merged['Rolling_Max_4w'] = train_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.rolling(window=4, min_periods=1).max())
test_merged['Rolling_Max_4w'] = np.nan

# Feature selection
feature_cols = ['Store', 'Dept', 'Size', 'Temperature', 'Fuel_Price', 'Total_MarkDown',
                'CPI', 'Unemployment', 'IsHoliday', 'Year', 'Month', 'Week', 'Day', 'Holiday_Dept',
                'Is_Peak_Week', 'Rolling_Max_4w'] + \
               [col for col in train_merged.columns if col.startswith('Type_')] + \
               [col for col in train_merged.columns if col.startswith('Lag_')]

X = train_merged[feature_cols]
y = train_merged['Weekly_Sales']
X_test = test_merged[feature_cols]

X = X.fillna(X.median(numeric_only=True))
X_test = X_test.fillna(X_test.median(numeric_only=True))


def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

dates = train_merged['Date']
unique_dates = sorted(dates.unique())
initial_train_period = 100
forecast_horizon = 13
metrics_list = []
val_data_all = pd.DataFrame()

for start in range(initial_train_period, len(unique_dates) - forecast_horizon, forecast_horizon):
    train_end_date = unique_dates[start]
    val_end_date = unique_dates[start + forecast_horizon]

    train_mask = dates <= train_end_date
    val_mask = (dates > train_end_date) & (dates <= val_end_date)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    if len(X_val) == 0 or len(y_val) == 0:
        continue

    peak_threshold = y_train.quantile(0.9)
    sample_weight = np.ones(len(y_train)) * 0.1
    sample_weight[y_train > peak_threshold] = 20.0

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        'max_depth': 15,
        'learning_rate': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.5,
        'seed': 42
    }

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=700,
        evals=[(dtrain, 'train'), (dval, 'eval')],
        early_stopping_rounds=10,
        verbose_eval=False,
        obj=custom_objective
    )

    xgb_preds = np.maximum(bst.predict(xgb.DMatrix(X_val)), 0)

    peak_mask = y_val > y_val.quantile(0.9)
    peak_mean = y_train[y_train > y_train.quantile(0.9)].mean()
    ensemble_preds = np.where(peak_mask, (xgb_preds * 0.7 + peak_mean * 0.3), xgb_preds)

    metrics_list.append({
        "RMSE": np.sqrt(mean_squared_error(y_val, ensemble_preds)),
        "R2": r2_score(y_val, ensemble_preds),
        "WMAPE": wmape(y_val, ensemble_preds)
    })

    fold_val_data = train_merged.loc[X_val.index].copy()
    fold_val_data['Predicted'] = ensemble_preds
    val_data_all = pd.concat([val_data_all, fold_val_data])

avg_metrics = pd.DataFrame(metrics_list).mean().to_dict()

avg_weekly_sales = y.mean()
rmse_percent = (avg_metrics['RMSE'] / avg_weekly_sales) * 100

print("=== XGBoost Rolling Origin Validation Metrics ===")
print(f"RMSE: {avg_metrics['RMSE']:.2f} ({rmse_percent:.2f}% of Avg Sales)")
print(f"R²: {avg_metrics['R2']:.4f}")
print(f"WMAPE: {avg_metrics['WMAPE']:.2f}%")


# PART 2  Random Forest (Improved Features)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('/content/drive/MyDrive/train.csv')
features_df = pd.read_csv('/content/drive/MyDrive/features.csv')
stores_df = pd.read_csv('/content/drive/MyDrive/stores.csv')

train_merged = train_df.merge(features_df, on=['Store', 'Date', 'IsHoliday'], how='left')
train_merged = train_merged.merge(stores_df, on='Store', how='left')

train_merged['Date'] = pd.to_datetime(train_merged['Date'])
train_merged = train_merged.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)

train_merged['Month'] = train_merged['Date'].dt.month
train_merged['Day'] = train_merged['Date'].dt.day
train_merged['WeekOfYear'] = train_merged['Date'].dt.isocalendar().week.astype(int)
train_merged['DayOfWeek'] = train_merged['Date'].dt.dayofweek
train_merged['IsHoliday'] = train_merged['IsHoliday'].astype(int)

train_merged['Season'] = train_merged['Month'].apply(
    lambda x: 1 if x in [12, 1, 2]
    else 2 if x in [3, 4, 5]
    else 3 if x in [6, 7, 8]
    else 4
)

lag_weeks = [1, 2, 3, 52]
for lag in lag_weeks:
    train_merged[f'Lag_{lag}'] = train_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)

train_merged['Rolling_Mean_3'] = train_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(3).mean()
train_merged['Rolling_Std_3'] = train_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(3).std()
train_merged['Rolling_Mean_8'] = train_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(8).mean()
train_merged['Rolling_Std_8'] = train_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(8).std()
train_merged['Rolling_Mean_12'] = train_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(12).mean()
train_merged['Rolling_Std_12'] = train_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(12).std()

train_merged['YoY_Lag_52_Ratio'] = train_merged['Rolling_Mean_12'] / (train_merged['Lag_52'].replace(0, np.nan))
train_merged['Store_Size_per_Dept'] = train_merged['Size'] / train_merged.groupby('Store')['Dept'].transform('nunique')
train_merged['Temp_Holiday_Interaction'] = train_merged['Temperature'] * train_merged['IsHoliday']

def add_holiday_distance_features(df, group_col='Store'):
    df = df.copy()
    prev_holiday = df.groupby(group_col).apply(lambda g: g['Date'].where(g['IsHoliday'] == 1).ffill()).reset_index(level=0, drop=True)
    df['Prev_Holiday_Date'] = prev_holiday
    next_holiday = df.groupby(group_col).apply(lambda g: g['Date'].where(g['IsHoliday'] == 1).bfill()).reset_index(level=0, drop=True)
    df['Next_Holiday_Date'] = next_holiday
    df['Days_Since_Last_Holiday'] = (df['Date'] - df['Prev_Holiday_Date']).dt.days
    df['Days_to_Next_Holiday'] = (df['Next_Holiday_Date'] - df['Date']).dt.days
    df['Days_Since_Last_Holiday'] = df['Days_Since_Last_Holiday'].fillna(999).astype(int)
    df['Days_to_Next_Holiday'] = df['Days_to_Next_Holiday'].fillna(999).astype(int)
    df = df.drop(columns=['Prev_Holiday_Date', 'Next_Holiday_Date'])
    return df

train_merged = add_holiday_distance_features(train_merged, group_col='Store')
train_merged = train_merged.dropna(subset=['Weekly_Sales'])
train_merged = train_merged[train_merged['Weekly_Sales'] >= 0]
train_merged['Weekly_Sales_Log'] = np.log1p(train_merged['Weekly_Sales'])

feature_cols = [
    'Store', 'Dept', 'Size', 'Temperature', 'Fuel_Price',
    'CPI', 'Unemployment', 'Month', 'Day', 'WeekOfYear', 'DayOfWeek',
    'IsHoliday', 'Season', 'Store_Size_per_Dept', 'Temp_Holiday_Interaction',
    'Days_Since_Last_Holiday', 'Days_to_Next_Holiday',
    'Rolling_Mean_3', 'Rolling_Std_3',
    'Rolling_Mean_8', 'Rolling_Std_8',
    'Rolling_Mean_12', 'Rolling_Std_12',
    'YoY_Lag_52_Ratio'
] + [f'Lag_{l}' for l in lag_weeks]

feature_cols = [c for c in feature_cols if c in train_merged.columns]
train_merged = train_merged.dropna(subset=feature_cols + ['Weekly_Sales_Log']).reset_index(drop=True)

X = train_merged[feature_cols]
y = train_merged['Weekly_Sales_Log']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred_log = rf.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

rmse_avg = np.sqrt(mean_squared_error(y_test_actual, y_pred))
rmse_percent = rmse_avg / np.mean(y_test_actual) * 100
rmse_per_sample = np.sqrt((y_test_actual.values - y_pred) ** 2).mean()
r2 = r2_score(y_test_actual, y_pred)
wmape = np.sum(np.abs(y_test_actual - y_pred)) / np.sum(np.abs(y_test_actual))

print("\n=== Random Forest Metrics ===")
print(f"R² Score: {r2:.4f}")
print(f"RMSE (average): {rmse_avg:.4f}")
print(f"RMSE (% of avg sales): {rmse_percent:.2f}%")
print(f"Per-sample RMSE (mean): {rmse_per_sample:.4f}")
print(f"WMAPE: {wmape:.4%}")

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 20 feature importances:")
print(importances.head(20))
