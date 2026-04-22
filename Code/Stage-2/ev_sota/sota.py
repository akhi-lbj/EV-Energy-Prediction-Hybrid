import pandas as pd
import numpy as np
import os
import warnings
import joblib
import wandb
from datetime import datetime
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import optuna
warnings.filterwarnings('ignore')

# ====================== W&B ======================
wandb.init(project="ev-charging-forecast-2026", name="phase2-sota-v2-fixed", config={"split": "50_10_40"})

print("🚀 EV Charging SOTA v2 - Phase 2 (Timezone Fixed + Full Features)")

# ====================== 1. LOAD & FIXED FEATURE ENGINEERING ======================
df = pd.read_csv("acn_enhanced_final_2019_data.csv")

# CRITICAL FIX: Make ALL datetimes timezone-aware (UTC) consistently
df['connectionTime'] = pd.to_datetime(df['connectionTime'], errors='coerce', utc=True)
df['disconnectTime'] = pd.to_datetime(df['disconnectTime'], errors='coerce', utc=True)
df['doneChargingTime'] = pd.to_datetime(df['doneChargingTime'], errors='coerce', utc=True)

df['hour'] = df['connectionTime'].dt.hour
df['month'] = df['connectionTime'].dt.month
df['day_of_week_encoded'] = pd.factorize(df['day_of_week'].fillna('Monday'))[0]
df['is_weekend'] = (df['day_of_week_encoded'] >= 5).astype(int)

# Duration (now both tz-aware)
df['duration_hours'] = (df['disconnectTime'] - df['connectionTime']).dt.total_seconds() / 3600

# Physics + behavioral features (your originals + 2025 SOTA)
df['charging_efficiency'] = df['kWhDelivered'] / df['parsed_kWhRequested'].replace(0, np.nan)
df['requested_gap'] = df['parsed_kWhRequested'] - df['kWhDelivered']
df['energy_per_minute'] = df['parsed_kWhRequested'] / (df['parsed_minutesAvailable'] + 1e-5)
df['request_efficiency'] = df['parsed_milesRequested'] / (df['parsed_WhPerMile'] + 1e-5)
df['urgency_flex_interaction'] = df['urgency_score'] * df['flexibility_index']
df['revision_urgency'] = df['revisionCount'] * df['urgency_score']
df['habit_grid'] = df['habit_stability'] * df['grid_impact_proxy'].fillna(1)
df['is_peak'] = df['hour'].between(17, 21).astype(int)

# Cyclic
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
print(f"✅ Clean dataset: {df.shape[0]:,} sessions | {len(feature_cols)} features")

# ====================== 2. 50:10:40 MONTH-STRATIFIED SPLIT ======================
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

print(f"Split → Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

# ====================== 3. TARGET ENCODING ======================
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
y_train = df_train[target]
X_val   = df_val[feature_cols].copy()
y_val   = df_val[target]
X_test  = df_test[feature_cols].copy()
y_test  = df_test[target]

# ====================== 4. OPTUNA + LIGHTGBM QUANTILE MODELS ======================
def objective(trial):
    params = {
        'objective': 'quantile',
        'alpha': 0.5,
        'metric': 'mae',
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(params, train_ds, num_boost_round=1200,
                      valid_sets=[val_ds], callbacks=[lgb.early_stopping(80, verbose=False)])
    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)

print("🔍 Running Optuna hyperparameter search (30 trials)...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
print(f"Best MAE: {study.best_value:.4f} | Params: {study.best_params}")

# Train final quantile models
quantiles = [0.05, 0.50, 0.95]
quantile_models = {}
os.makedirs("sota_models_v2", exist_ok=True)

for alpha in quantiles:
    params = {**study.best_params, 'objective': 'quantile', 'alpha': alpha, 'metric': 'quantile'}
    train_ds = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_ds, num_boost_round=1500,
                      valid_sets=[lgb.Dataset(X_val, label=y_val)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    quantile_models[alpha] = model
    model.save_model(f"sota_models_v2/lgb_quantile_alpha_{int(alpha*100)}.txt")

# ====================== 5. EVALUATION ======================
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"📊 {name:6} → MAE: {mae:.3f} kWh | RMSE: {rmse:.3f} | R²: {r2:.4f}")
    return mae, rmse, r2

pred_test = quantile_models[0.50].predict(X_test.values if hasattr(X_test, 'values') else X_test)

train_mae, train_rmse, train_r2 = evaluate(y_train, quantile_models[0.50].predict(X_train), "Train")
val_mae,   val_rmse,   val_r2   = evaluate(y_val,   quantile_models[0.50].predict(X_val),   "Val")
test_mae,  test_rmse,  test_r2  = evaluate(y_test,  pred_test, "Test")

# Probabilistic Interval
lower = quantile_models[0.05].predict(X_test)
upper = quantile_models[0.95].predict(X_test)
coverage = np.mean((y_test >= lower) & (y_test <= upper))
width = np.mean(upper - lower)
print(f"🎯 90% PI → Coverage: {coverage:.1%} | Avg Width: {width:.3f} kWh")

# W&B logging
wandb.log({
    "test_mae": test_mae, "test_rmse": test_rmse, "test_r2": test_r2,
    "pi_coverage_90": coverage, "pi_width": width,
    "best_optuna_mae": study.best_value
})

# Save encoders
joblib.dump(te_station, "sota_models_v2/te_station.pkl")
joblib.dump(te_user, "sota_models_v2/te_user.pkl")

print("\n✅ Phase 2 COMPLETE & FIXED! Models saved in 'sota_models_v2/'")
wandb.finish()