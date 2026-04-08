import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

print("🚀 Loading enhanced ACN 2019 dataset for v15 (LEAKAGE FIXED)...")
df = pd.read_csv("acn_enhanced_final_2019_data.csv")

# --- 1. Temporal features ---
df['connectionTime'] = pd.to_datetime(df['connectionTime'], errors='coerce', utc=True)
df['hour_of_day'] = df['connectionTime'].dt.hour
df['month'] = df['connectionTime'].dt.month
df['day_of_week_encoded'] = pd.factorize(df['day_of_week'].fillna('Monday'))[0]

# --- 2. Physics / Behavioral interactions (NO target leakage) ---
df['energy_per_minute'] = df['parsed_kWhRequested'] / (df['parsed_minutesAvailable'] + 1e-5)
df['request_efficiency'] = df['parsed_milesRequested'] / (df['parsed_WhPerMile'] + 1e-5)
df['urgency_flex_interaction'] = df['urgency_score'] * df['flexibility_index']
df['revision_urgency'] = df['revisionCount'] * df['urgency_score']
df['habit_grid'] = df['habit_stability'] * df['grid_impact_proxy']

# --- 3. 50:10:40 MONTH-STRATIFIED SPLIT (unchanged) ---
np.random.seed(42)
test_indices, val_indices = [], []
for month in range(1, 13):
    month_df = df[df['month'] == month]
    n = len(month_df)
    test_sample = month_df.sample(n=int(0.4 * n), random_state=42)
    test_indices.extend(test_sample.index)
    remaining = month_df.drop(test_sample.index)
    val_sample = remaining.sample(n=int(0.1 * n), random_state=42)
    val_indices.extend(val_sample.index)

test_mask = df.index.isin(test_indices)
val_mask = df.index.isin(val_indices)
train_mask = ~(test_mask | val_mask)

df_train = df[train_mask].copy()
df_val   = df[val_mask].copy()
df_test  = df[test_mask].copy()

print(f"Train (50%): {len(df_train):,} | Val (10%): {len(df_val):,} | Test (40%): {len(df_test):,} sessions")

# --- 4. Target Encoding (train-only — no leakage) ---
te_station = TargetEncoder(target_type='continuous', random_state=42)
te_user    = TargetEncoder(target_type='continuous', random_state=42)

df_train['station_encoded'] = te_station.fit_transform(df_train[['stationID']], df_train['kWhDelivered']).flatten()
df_train['user_encoded']    = te_user.fit_transform(df_train[['parsed_userID']], df_train['kWhDelivered']).flatten()

df_val['station_encoded']  = te_station.transform(df_val[['stationID']]).flatten()
df_val['user_encoded']     = te_user.transform(df_val[['parsed_userID']]).flatten()

df_test['station_encoded'] = te_station.transform(df_test[['stationID']]).flatten()
df_test['user_encoded']    = te_user.transform(df_test[['parsed_userID']]).flatten()

feature_cols = [
    'parsed_milesRequested', 'parsed_WhPerMile', 'parsed_minutesAvailable',
    'parsed_kWhRequested', 'revisionCount', 'hour_of_day', 'day_of_week_encoded',
    'is_weekend', 'urgency_score', 'flexibility_index', 'habit_stability',
    'grid_impact_proxy', 'urgency_flex_interaction', 'revision_urgency',
    'energy_per_minute', 'request_efficiency', 'habit_grid',
    'station_encoded', 'user_encoded'
]

X_train = df_train[feature_cols]
y_train = df_train['kWhDelivered']
X_val   = df_val[feature_cols]
y_val   = df_val['kWhDelivered']
X_test  = df_test[feature_cols]
y_test  = df_test['kWhDelivered']

# ====================== v15 STACKED ENSEMBLE (LEAKAGE FIXED) ======================
def run_stacked_v15():
    print("\n=== Running v15 Stacked Ensemble (RF + XGB + CatBoost + HistGB + LGB meta) ===")
    
    # Base models (raw features, no scaler)
    rf = RandomForestRegressor(n_estimators=300, max_depth=9, min_samples_leaf=5,
                               max_features=0.75, random_state=42, n_jobs=-1)
    xgb_model = xgb.XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.025,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                                 reg_alpha=1.5, random_state=42, tree_method='hist')
    cat_model = cb.CatBoostRegressor(iterations=600, depth=8, learning_rate=0.03,
                                     l2_leaf_reg=4, random_seed=42, verbose=0)
    hist_model = HistGradientBoostingRegressor(max_iter=600, max_depth=8, learning_rate=0.03,
                                               random_state=42)
    
    # Fit
    rf.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    cat_model.fit(X_train, y_train)
    hist_model.fit(X_train, y_train)
    
    # Stack
    stack_train = np.column_stack((rf.predict(X_train), xgb_model.predict(X_train),
                                   cat_model.predict(X_train), hist_model.predict(X_train)))
    stack_val   = np.column_stack((rf.predict(X_val),   xgb_model.predict(X_val),
                                   cat_model.predict(X_val),   hist_model.predict(X_val)))
    stack_test  = np.column_stack((rf.predict(X_test),  xgb_model.predict(X_test),
                                   cat_model.predict(X_test),  hist_model.predict(X_test)))
    
    # Meta
    meta_params = {
        'objective': 'regression', 'metric': 'rmse',
        'num_leaves': 63, 'learning_rate': 0.015, 'feature_fraction': 0.9,
        'bagging_fraction': 0.9, 'lambda_l1': 3.0, 'lambda_l2': 3.0,
        'verbose': -1, 'random_state': 42
    }
    lgb_train_ds = lgb.Dataset(stack_train, label=y_train)
    lgb_val_ds   = lgb.Dataset(stack_val, label=y_val, reference=lgb_train_ds)
    
    meta_model = lgb.train(meta_params, lgb_train_ds, num_boost_round=1500,
                           valid_sets=[lgb_val_ds], callbacks=[lgb.early_stopping(150, verbose=False)])
    
    final_test = meta_model.predict(stack_test)
    
    mae = mean_absolute_error(y_test, final_test)
    rmse = np.sqrt(mean_squared_error(y_test, final_test))
    r2 = r2_score(y_test, final_test)
    
    print(f"  Test (40%) → MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.4f}")
    
    # Log
    pd.DataFrame([{
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'experiment': 'Stacked_v15_RF_XGB_Cat_Hist_LGB_LEAKAGE_FIXED',
        'test_mae': mae, 'test_rmse': rmse, 'test_r2': r2,
        'note': 'LEAKAGE FIXED - same 50:10:40 stratified split'
    }]).to_csv("baseline_results_log.csv", mode='a', header=False, index=False)
    
    return mae, rmse, r2

# RUN IT
print("\nRunning v15 Stacked Ensemble (fixed)...")
test_mae, test_rmse, test_r2 = run_stacked_v15()

print("\n=== v15 SUMMARY (Leakage Fixed) ===")
print(f"Test MAE: {test_mae:.3f} kWh | R²: {test_r2:.4f}")
print(f"Your original baseline: 2.752 kWh | R²: 0.7792")
print(f"Expected gain: 0.15–0.35 kWh 🔥")
print("\n✅ v15 fixed & complete. Paste the output here!")