import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ====================== 1. LOAD & PREPROCESS ======================
print("🚀 Loading enhanced ACN 2019 dataset for Improved Stacked Ensemble...")
df = pd.read_csv("acn_enhanced_final_2019_data.csv")

df['connectionTime'] = pd.to_datetime(df['connectionTime'], errors='coerce', utc=True)
df['hour_of_day'] = df['connectionTime'].dt.hour
df['month'] = df['connectionTime'].dt.month

day_le = LabelEncoder()
df['day_of_week_encoded'] = day_le.fit_transform(df['day_of_week'].fillna('Monday'))

grid_le = LabelEncoder()
df['grid_impact_proxy_encoded'] = grid_le.fit_transform(df['grid_impact_proxy'].fillna('medium'))

df['urgency_flex_interaction'] = df['urgency_score'] * df['flexibility_index']
df['revision_urgency'] = df['revisionCount'] * df['urgency_score']

feature_cols = [
    'parsed_milesRequested', 'parsed_WhPerMile', 'parsed_minutesAvailable',
    'parsed_kWhRequested', 'revisionCount', 'hour_of_day', 
    'day_of_week_encoded', 'is_weekend',
    'urgency_score', 'flexibility_index', 'habit_stability',
    'grid_impact_proxy_encoded', 'urgency_flex_interaction', 'revision_urgency'
]

target = 'kWhDelivered'
df = df.dropna(subset=[target] + feature_cols).copy()

# ====================== 2. STRATIFIED BALANCED SPLIT ======================
np.random.seed(42)
test_indices = []
val_indices = []

for month in range(1, 13):
    month_df = df[df['month'] == month]
    n_month = len(month_df)
    
    test_sample = month_df.sample(n=min(67, n_month), random_state=42)
    test_indices.extend(test_sample.index)
    
    remaining = month_df.drop(test_sample.index)
    val_sample = remaining.sample(n=min(50, len(remaining)), random_state=42)
    val_indices.extend(val_sample.index)

test_mask = df.index.isin(test_indices)
val_mask = df.index.isin(val_indices)
train_mask = ~(test_mask | val_mask)

X = df[feature_cols]
y = df[target]

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"Train: {len(X_train):,} | Val (~50/month): {len(X_val):,} | Test (~67/month): {len(X_test):,}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ====================== 3. ABLATION: Run Twice (With vs Without Behavioral) ======================
def run_stacked(with_behavioral=True):
    suffix = "With_Behavioral" if with_behavioral else "Without_Behavioral"
    print(f"\n=== Running Stacked Ensemble: {suffix} ===")
    
    cols = feature_cols.copy()
    if not with_behavioral:
        cols = [c for c in cols if c not in ['urgency_score', 'flexibility_index', 
                                             'habit_stability', 'grid_impact_proxy_encoded',
                                             'urgency_flex_interaction', 'revision_urgency']]
    
    X_train_cur = X_train[cols] if with_behavioral else X_train_s[:, [feature_cols.index(c) for c in cols]]
    X_val_cur   = X_val[cols] if with_behavioral else X_val_s[:, [feature_cols.index(c) for c in cols]]
    X_test_cur  = X_test[cols] if with_behavioral else X_test_s[:, [feature_cols.index(c) for c in cols]]
    
    # Re-scale if needed
    if not with_behavioral:
        scaler_cur = StandardScaler()
        X_train_cur = scaler_cur.fit_transform(X_train_cur)
        X_val_cur   = scaler_cur.transform(X_val_cur)
        X_test_cur  = scaler_cur.transform(X_test_cur)
    else:
        X_train_cur = X_train_s
        X_val_cur   = X_val_s
        X_test_cur  = X_test_s
    
    # Level-0 Base Models (Regularized)
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=4, 
                               random_state=42, n_jobs=-1)
    rf.fit(X_train_cur, y_train)
    
    xgb_model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.04,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                 random_state=42, tree_method='hist')
    xgb_model.fit(X_train_cur, y_train)
    
    knn = KNeighborsRegressor(n_neighbors=20, weights='distance', n_jobs=-1)
    knn.fit(X_train_cur, y_train)
    
    # Level-0 predictions
    rf_train = rf.predict(X_train_cur)
    xgb_train = xgb_model.predict(X_train_cur)
    knn_train = knn.predict(X_train_cur)
    
    rf_val = rf.predict(X_val_cur)
    xgb_val = xgb_model.predict(X_val_cur)
    knn_val = knn.predict(X_val_cur)
    
    rf_test = rf.predict(X_test_cur)
    xgb_test = xgb_model.predict(X_test_cur)
    knn_test = knn.predict(X_test_cur)
    
    stack_train = np.column_stack((rf_train, xgb_train, knn_train))
    stack_val   = np.column_stack((rf_val, xgb_val, knn_val))
    stack_test  = np.column_stack((rf_test, xgb_test, knn_test))
    
    # Level-1 Meta (Regularized LightGBM)
    meta_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 20,
        'learning_rate': 0.03,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'verbose': -1,
        'random_state': 42
    }
    
    lgb_train_ds = lgb.Dataset(stack_train, label=y_train)
    lgb_val_ds   = lgb.Dataset(stack_val, label=y_val, reference=lgb_train_ds)
    
    meta_model = lgb.train(
        meta_params, lgb_train_ds, num_boost_round=800,
        valid_sets=[lgb_val_ds], callbacks=[lgb.early_stopping(80, verbose=False)]
    )
    
    # Final predictions
    final_train = meta_model.predict(stack_train)
    final_val   = meta_model.predict(stack_val)
    final_test  = meta_model.predict(stack_test)
    
    # Evaluate
    def eval_set(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"  {name} → MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.4f}")
        return mae, rmse, r2
    
    print(f"\n{suffix} Results:")
    eval_set(y_train, final_train, "Train")
    eval_set(y_val,   final_val,   "Validation")
    test_mae, test_rmse, test_r2 = eval_set(y_test, final_test, "Test (Balanced)")
    
    # Log
    results_row = pd.DataFrame([{
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'experiment': f'Improved_Stacked_{suffix}',
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'note': 'Regularized bases + meta'
    }])
    
    log_file = "baseline_results_log.csv"
    if os.path.exists(log_file):
        results_row.to_csv(log_file, mode='a', header=False, index=False)
    else:
        results_row.to_csv(log_file, index=False)
    
    return test_mae, test_rmse, test_r2

# Run both ablations
print("\nRunning Ablation Study...")
with_mae, with_rmse, with_r2 = run_stacked(with_behavioral=True)
without_mae, without_rmse, without_r2 = run_stacked(with_behavioral=False)

print("\n=== ABLATION SUMMARY ===")
print(f"With Behavioral Features  → Test MAE: {with_mae:.3f} kWh | R²: {with_r2:.4f}")
print(f"Without Behavioral Features → Test MAE: {without_mae:.3f} kWh | R²: {without_r2:.4f}")
print(f"Gain from Behavioral Features: {without_mae - with_mae:.3f} kWh improvement in MAE")

print("\n✅ Improved Stacked Ensemble + Ablation completed.")
print("Check baseline_results_log.csv for all runs.")

# Save final models (With behavioral version)
os.makedirs("models", exist_ok=True)
# (You can extend saving if needed)
print("Models can be saved similarly as before if required.")