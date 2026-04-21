import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import os
import warnings
import joblib
from datetime import datetime
warnings.filterwarnings('ignore')

# ====================== 1. LOAD & PREPROCESS ======================
print("🚀 Loading enhanced ACN 2019 dataset...")
df = pd.read_csv("acn_enhanced_final_2019_data.csv")

df['connectionTime'] = pd.to_datetime(df['connectionTime'], errors='coerce', utc=True)
df['disconnectTime'] = pd.to_datetime(df['disconnectTime'], errors='coerce', utc=True)

df['hour'] = df['connectionTime'].dt.hour
df['month'] = df['connectionTime'].dt.month
df['day_of_week_encoded'] = pd.factorize(df['day_of_week'].fillna('Monday'))[0]
df['is_weekend'] = (df['day_of_week_encoded'] >= 5).astype(int)
df['duration_hours'] = (df['disconnectTime'] - df['connectionTime']).dt.total_seconds() / 3600
df['charging_efficiency'] = df['kWhDelivered'] / df['parsed_kWhRequested'].replace(0, np.nan)
df['requested_gap'] = df['parsed_kWhRequested'] - df['kWhDelivered']
df['energy_per_minute'] = df['parsed_kWhRequested'] / (df['parsed_minutesAvailable'] + 1e-5)
df['request_efficiency'] = df['parsed_milesRequested'] / (df['parsed_WhPerMile'] + 1e-5)
df['urgency_flex_interaction'] = df['urgency_score'] * df['flexibility_index']
df['revision_urgency'] = df['revisionCount'] * df['urgency_score']
df['habit_grid'] = df['habit_stability'] * df['grid_impact_proxy'].fillna(1)
df['is_peak'] = df['hour'].between(17, 21).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Define Feature Sets (aligned with SOTA ablation)
behavioral_features = ['urgency_flex_interaction', 'revision_urgency', 'habit_grid', 'energy_per_minute']
base_features = [
    'parsed_milesRequested', 'parsed_WhPerMile', 'parsed_minutesAvailable', 'parsed_kWhRequested',
    'revisionCount', 'hour', 'day_of_week_encoded', 'is_weekend', 'is_peak',
    'urgency_score', 'flexibility_index', 'habit_stability', 'grid_impact_proxy',
    'request_efficiency', 'duration_hours', 'charging_efficiency', 'requested_gap',
    'hour_sin', 'hour_cos'
]

target = 'kWhDelivered'

# Clean data
df = df.dropna(subset=[target] + base_features + behavioral_features).copy()
print(f"✅ Clean dataset: {df.shape[0]:,} sessions")

# ====================== 2. 50:10:40 MONTH-STRATIFIED SPLIT ======================
np.random.seed(42)
test_indices, val_indices = [], []

for month in range(1, 13):
    month_df = df[df['month'] == month]
    n_month = len(month_df)
    test_sample = month_df.sample(n=int(0.4 * n_month), random_state=42)
    test_indices.extend(test_sample.index)
    remaining = month_df.drop(test_sample.index)
    val_sample = remaining.sample(n=int(0.1 * n_month), random_state=42)
    val_indices.extend(val_sample.index)

test_mask = df.index.isin(test_indices)
val_mask = df.index.isin(val_indices)
train_mask = ~(test_mask | val_mask)

# ====================== 3. EXPERIMENT FUNCTION ======================
def run_experiment(use_behavioral=True):
    suffix = "WITH_BEHAVIORAL" if use_behavioral else "WITHOUT_BEHAVIORAL"
    print(f"\n--- 🚀 Running Experiment: {suffix} ---")
    
    current_features = base_features + (behavioral_features if use_behavioral else [])
    X_train, y_train = df.loc[train_mask, current_features], df.loc[train_mask, target]
    X_val,   y_val   = df.loc[val_mask,   current_features], df.loc[val_mask,   target]
    X_test,  y_test  = df.loc[test_mask,  current_features], df.loc[test_mask,  target]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dval   = xgb.DMatrix(X_val_scaled,   label=y_val)
    dtest  = xgb.DMatrix(X_test_scaled,  label=y_test)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': ['rmse', 'mae'],
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'random_state': 42,
        'tree_method': 'hist'
    }

    def xg_r2_score(preds, dmatrix):
        labels = dmatrix.get_label()
        return 'r2', r2_score(labels, preds)

    evals_result = {}
    model = xgb.train(
        params, dtrain, num_boost_round=1200,
        evals=[(dtrain, 'train'), (dval, 'val')],
        custom_metric=xg_r2_score,
        evals_result=evals_result,
        early_stopping_rounds=60,
        verbose_eval=False
    )

    # Evaluation
    def evaluate(m, X_s, y_t):
        preds = m.predict(xgb.DMatrix(X_s))
        return mean_absolute_error(y_t, preds), np.sqrt(mean_squared_error(y_t, preds)), r2_score(y_t, preds)

    tr_mae, tr_rmse, tr_r2 = evaluate(model, X_train_scaled, y_train)
    vl_mae, vl_rmse, vl_r2 = evaluate(model, X_val_scaled, y_val)
    ts_mae, ts_rmse, ts_r2 = evaluate(model, X_test_scaled, y_test)

    print(f"📊 {suffix} Results:")
    print(f"  Test MAE : {ts_mae:.3f} kWh | RMSE: {ts_rmse:.3f} | R²: {ts_r2:.4f}")

    # Log results
    log_file = "baseline_results_log_v2.csv"
    results_row = pd.DataFrame([{
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'experiment': suffix,
        'n_features': len(current_features),
        'test_mae': ts_mae, 'test_rmse': ts_rmse, 'test_r2': ts_r2
    }])
    results_row.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

    # Save Model Artifacts
    model_dir = f"xgboost_models_{suffix}"
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(f"{model_dir}/xgb_model.json")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")
    
    return ts_mae, ts_r2

# ====================== 4. EXECUTION ======================
mae_with, r2_with = run_experiment(use_behavioral=True)
mae_without, r2_without = run_experiment(use_behavioral=False)

print("\n=== FINAL ABLATION SUMMARY (XGBoost Baseline) ===")
print(f"WITH Behavioral    : MAE {mae_with:.3f} | R² {r2_with:.4f}")
print(f"WITHOUT Behavioral : MAE {mae_without:.3f} | R² {r2_without:.4f}")
print(f"Impact of Behavioral features → MAE Delta: {mae_without - mae_with:.3f}")

print("\n✅ Baseline ablation study complete.")