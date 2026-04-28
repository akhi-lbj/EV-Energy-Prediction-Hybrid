import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# Load saved models from Phase 4
rf = joblib.load("sota_models_v3/rf_base.pkl")
xgb_model = joblib.load("sota_models_v3/xgb_base.pkl")
cat_model = joblib.load("sota_models_v3/cat_base.pkl")
lgb_base = joblib.load("sota_models_v3/lgb_base.pkl")
meta_model = lgb.Booster(model_file="sota_models_v3/meta_lgb.txt")

# Load encoders
te_station = joblib.load("sota_models_v2/te_station.pkl")
te_user = joblib.load("sota_models_v2/te_user.pkl")

# Reload test set (same preprocessing as before)
df = pd.read_csv("acn_enhanced_final_2019_data.csv")
# ... (copy the exact feature engineering from Phase 4 here - abbreviated)

# Prepare X_test same as Phase 4
# X_test = ...
df = pd.read_csv("acn_enhanced_final_2019_data.csv")
df['connectionTime'] = pd.to_datetime(df['connectionTime'], errors='coerce', utc=True)
df['disconnectTime'] = pd.to_datetime(df['disconnectTime'], errors='coerce', utc=True)

# Features (same robust set)
df['hour'] = df['connectionTime'].dt.hour
df['month'] = df['connectionTime'].dt.month
df['day_of_week_encoded'] = pd.factorize(df['day_of_week'].fillna('Monday'))[0]
df['is_weekend'] = (df['day_of_week_encoded'] >= 5).astype(int)
df['duration_hours'] = (df['disconnectTime'] - df['connectionTime']).dt.total_seconds() / 3600
from _internal_features import add_advanced_features
df = add_advanced_features(df)
df['energy_per_minute'] = df['parsed_kWhRequested'] / (df['parsed_minutesAvailable'] + 1e-5)
df['request_efficiency'] = df['parsed_milesRequested'] / (df['parsed_WhPerMile'] + 1e-5)
df['urgency_flex_interaction'] = df['urgency_score'] * df['flexibility_index']
df['revision_urgency'] = df['revisionCount'] * df['urgency_score']
df['habit_grid'] = df['habit_stability'] * df['grid_impact_proxy'].fillna(1)
df['is_peak'] = df['hour'].between(17, 21).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

feature_cols = ['parsed_milesRequested', 'parsed_WhPerMile', 'parsed_minutesAvailable', 'parsed_kWhRequested',
                'revisionCount', 'hour', 'day_of_week_encoded', 'is_weekend', 'is_peak',
                'urgency_score', 'flexibility_index', 'habit_stability', 'grid_impact_proxy',
                'urgency_flex_interaction', 'revision_urgency', 'habit_grid', 'energy_per_minute',
                'request_efficiency', 'duration_hours', 'charging_efficiency', 'requested_gap',
                'hour_sin', 'hour_cos']

target = 'kWhDelivered'
df = df.dropna(subset=[target] + feature_cols).copy()

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
df_val = df[val_mask].copy()
df_test = df[test_mask].copy()


df_train['station_encoded'] = te_station.fit_transform(df_train[['stationID']], df_train[target]).flatten()
df_train['user_encoded'] = te_user.fit_transform(df_train[['parsed_userID']], df_train[target]).flatten()
df_val['station_encoded'] = te_station.transform(df_val[['stationID']]).flatten()
df_val['user_encoded'] = te_user.transform(df_val[['parsed_userID']]).flatten()
df_test['station_encoded'] = te_station.transform(df_test[['stationID']]).flatten()
df_test['user_encoded'] = te_user.transform(df_test[['parsed_userID']]).flatten()

feature_cols += ['station_encoded', 'user_encoded']

X_train = df_train[feature_cols].copy()
y_train = df_train[target].copy()
X_val = df_val[feature_cols].copy()
y_val = df_val[target].copy()
X_test = df_test[feature_cols].copy()
y_test = df_test[target].copy()

# 1. Point forecast from stacking
stack_test = np.column_stack((
    rf.predict(X_test), 
    xgb_model.predict(X_test), 
    cat_model.predict(X_test), 
    lgb_base.predict(X_test)
))
point_pred = meta_model.predict(stack_test)

# 2. Probabilistic: Load or retrain quantile models (best from Phase 3)
quantile_models = {}
for alpha in [0.05, 0.50, 0.95]:
    model = lgb.Booster(model_file=f"sota_models_v2/lgb_quantile_alpha_{int(alpha*100)}.txt")
    quantile_models[alpha] = model

q05 = quantile_models[0.05].predict(X_test)
q50 = quantile_models[0.50].predict(X_test)
q95 = quantile_models[0.95].predict(X_test)

# 3. Conformal adjustment (from Phase 3)
q_hat = 0.629  # from your Phase 3 run - or recompute on val set
lower = q05 - q_hat
upper = q95 + q_hat

# ====================== PROBABILISTIC METRICS ======================
print("=== FULL PROBABILISTIC FORECASTING RESULTS ===")
print(f"Point Forecast (Stack Meta) → Test MAE: {mean_absolute_error(y_test, point_pred):.3f} kWh")

def pi_metrics(y_true, lower, upper, nominal=0.90):
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    width = np.mean(upper - lower)
    print(f"{nominal*100}% PI Coverage: {coverage:.1%} | Average Width: {width:.3f} kWh")
    return coverage, width

pi_metrics(y_test, lower, upper)

# Example inference with full distribution
print("\n🔍 Example Prediction with Uncertainty:")
example_idx = 0
print(f"Session {example_idx}:")
print("  Input Features:")
for col, val in X_test.iloc[example_idx].items():
    print(f"    {col}: {val}")
print(f"  Predicted (median): {q50[example_idx]:.2f} kWh")
print(f"  90% PI: [{lower[example_idx]:.2f}, {upper[example_idx]:.2f}] kWh")
print(f"  True value: {y_test.iloc[example_idx]:.2f} kWh")

# Save full probabilistic predictions
prob_df = pd.DataFrame({
    'true_kWh': y_test,
    'pred_median': q50,
    'pred_05': q05,
    'pred_95': q95,
    'lower_conformal': lower,
    'upper_conformal': upper
})
prob_df.to_csv("probabilistic_predictions_test.csv", index=False)
print("✅ Full probabilistic predictions saved to 'probabilistic_predictions_test.csv'")