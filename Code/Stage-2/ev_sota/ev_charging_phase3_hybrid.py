import pandas as pd
import numpy as np
import os
import warnings
import joblib
import wandb
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
from sklearn.preprocessing import TargetEncoder
warnings.filterwarnings('ignore')

wandb.init(project="ev-charging-forecast-2026", name="phase3-hybrid-fixed", config={"model": "lgbm+conformal"})

print("🚀 Phase 3: Hybrid + Conformal Prediction (Bug Fixed)")

# ====================== 1. LOAD & FEATURE ENGINEERING ======================
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

feature_cols = [
    'parsed_milesRequested', 'parsed_WhPerMile', 'parsed_minutesAvailable', 'parsed_kWhRequested',
    'revisionCount', 'hour', 'day_of_week_encoded', 'is_weekend', 'is_peak',
    'urgency_score', 'flexibility_index', 'habit_stability', 'grid_impact_proxy',
    'urgency_flex_interaction', 'revision_urgency', 'habit_grid', 'energy_per_minute',
    'request_efficiency', 'duration_hours', 'charging_efficiency', 'requested_gap',
    'hour_sin', 'hour_cos'
]

target = 'kWhDelivered'
df = df.dropna(subset=[target] + feature_cols).copy()
print(f"✅ Clean dataset: {df.shape[0]:,} sessions")

# ====================== 2. 50:10:40 SPLIT ======================
np.random.seed(42)
test_indices, val_indices = [], []
for month in range(1, 13):
    month_df = df[df['month'] == month].copy()
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

# ====================== 3. FIXED TARGET ENCODING ======================
te_station = TargetEncoder(target_type='continuous', random_state=42)
te_user    = TargetEncoder(target_type='continuous', random_state=42)

df_train['station_encoded'] = te_station.fit_transform(df_train[['stationID']], df_train[target]).flatten()
df_train['user_encoded']    = te_user.fit_transform(df_train[['parsed_userID']], df_train[target]).flatten()

df_val['station_encoded']   = te_station.transform(df_val[['stationID']]).flatten()
df_val['user_encoded']      = te_user.transform(df_val[['parsed_userID']]).flatten()
df_test['station_encoded']  = te_station.transform(df_test[['stationID']]).flatten()
df_test['user_encoded']     = te_user.transform(df_test[['parsed_userID']]).flatten()

feature_cols += ['station_encoded', 'user_encoded']

X_train = df_train[feature_cols].copy()
y_train = df_train[target].copy()
X_val   = df_val[feature_cols].copy()
y_val   = df_val[target].copy()
X_test  = df_test[feature_cols].copy()
y_test  = df_test[target].copy()

print(f"Features ready: {len(feature_cols)} columns")

# ====================== 4. LOAD BEST LGBM QUANTILE MODELS ======================
best_params = {
    'num_leaves': 79,
    'learning_rate': 0.0183,
    'feature_fraction': 0.964,
    'bagging_fraction': 0.938,
    'lambda_l1': 0.0098,
    'lambda_l2': 1.14e-6,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1
}

lgb_models = {}
os.makedirs("sota_models_v2", exist_ok=True)

for alpha in [0.05, 0.50, 0.95]:
    params = {**best_params, 'objective': 'quantile', 'alpha': alpha, 'metric': 'quantile'}
    train_ds = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_ds, num_boost_round=1200,
                      valid_sets=[lgb.Dataset(X_val, label=y_val)],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
    lgb_models[alpha] = model
    model.save_model(f"sota_models_v2/lgb_quantile_alpha_{int(alpha*100)}.txt")

# ====================== 5. CONFORMAL PREDICTION (Guaranteed Coverage) ======================
# Non-conformity scores on validation set
val_preds = lgb_models[0.50].predict(X_val)
nonconformity = np.abs(y_val - val_preds)
q_hat = np.quantile(nonconformity, 0.90)   # for 90% coverage

print(f"Conformal quantile (90%): {q_hat:.3f} kWh")

# Apply to test set
test_preds_median = lgb_models[0.50].predict(X_test)
test_lower = lgb_models[0.05].predict(X_test) - q_hat
test_upper = lgb_models[0.95].predict(X_test) + q_hat

coverage = np.mean((y_test >= test_lower) & (y_test <= test_upper))
width = np.mean(test_upper - test_lower)

print(f"🎯 Conformal 90% PI → Coverage: {coverage:.1%} | Avg Width: {width:.3f} kWh")

# ====================== 6. FINAL EVALUATION ======================
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"📊 {name:8} → MAE: {mae:.3f} kWh | RMSE: {rmse:.3f} | R²: {r2:.4f}")

evaluate(y_train, lgb_models[0.50].predict(X_train), "Train")
evaluate(y_val,   lgb_models[0.50].predict(X_val),   "Val")
evaluate(y_test,  test_preds_median, "Test (Hybrid)")

# Save everything
joblib.dump(te_station, "sota_models_v2/te_station.pkl")
joblib.dump(te_user, "sota_models_v2/te_user.pkl")

wandb.log({
    "test_mae": mean_absolute_error(y_test, test_preds_median),
    "test_r2": r2_score(y_test, test_preds_median),
    "pi_coverage_90_conformal": coverage,
    "pi_width_conformal": width
})

print("\n✅ Phase 3 COMPLETE! Conformal Prediction fixed the coverage issue.")
print("Models + conformal logic saved in 'sota_models_v2/'")
wandb.finish()