import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ====================== 1. LOAD & PREPROCESS ======================
print("🚀 Loading enhanced ACN 2019 dataset...")
df = pd.read_csv("acn_enhanced_final_2019_data.csv")

df['connectionTime'] = pd.to_datetime(df['connectionTime'], errors='coerce', utc=True)
df['hour_of_day'] = df['connectionTime'].dt.hour
df['month'] = df['connectionTime'].dt.month

# Encode categoricals
day_le = LabelEncoder()
df['day_of_week_encoded'] = day_le.fit_transform(df['day_of_week'].fillna('Monday'))

grid_le = LabelEncoder()
df['grid_impact_proxy_encoded'] = grid_le.fit_transform(df['grid_impact_proxy'].fillna('medium'))

# Behavioral interactions
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
print(f"✅ Clean dataset: {df.shape[0]:,} sessions | {len(feature_cols)} features")

# ====================== 2. 50:10:40 MONTH-STRATIFIED SPLIT (EXACTLY matching v15/v16) ======================
np.random.seed(42)
test_indices, val_indices = [], []

for month in range(1, 13):
    month_df = df[df['month'] == month]
    n_month = len(month_df)
    
    # Test = 40%
    test_sample = month_df.sample(n=int(0.4 * n_month), random_state=42)
    test_indices.extend(test_sample.index)
    
    remaining = month_df.drop(test_sample.index)
    
    # Validation = 10%
    val_sample = remaining.sample(n=int(0.1 * n_month), random_state=42)
    val_indices.extend(val_sample.index)

test_mask = df.index.isin(test_indices)
val_mask = df.index.isin(val_indices)
train_mask = ~(test_mask | val_mask)

X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, target]
X_val,   y_val   = df.loc[val_mask, feature_cols],   df.loc[val_mask, target]
X_test,  y_test  = df.loc[test_mask, feature_cols],  df.loc[test_mask, target]

print(f"Train (50%): {len(X_train):,} sessions")
print(f"Validation (10%): {len(X_val):,} sessions")
print(f"Test (40%): {len(X_test):,} sessions (all months present)")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dval   = xgb.DMatrix(X_val_scaled,   label=y_val)
dtest  = xgb.DMatrix(X_test_scaled,  label=y_test)

# ====================== 3. TRAIN XGBoost ======================
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

print("\nTraining XGBoost Baseline...")

def xg_r2_score(preds, dmatrix):
    labels = dmatrix.get_label()
    r2 = r2_score(labels, preds)
    return 'r2', r2

evals_result = {}

model = xgb.train(
    params, dtrain, num_boost_round=1200,
    evals=[(dtrain, 'train'), (dval, 'val')],
    custom_metric=xg_r2_score,
    evals_result=evals_result,
    early_stopping_rounds=60,
    verbose_eval=100
)

# --- Save Per-Iteration Metrics ---
iterations = len(evals_result['train']['rmse'])
history_df = pd.DataFrame({
    'iteration': range(1, iterations + 1),
    'train_rmse': evals_result['train']['rmse'],
    'train_mae': evals_result['train']['mae'],
    'train_r2': evals_result['train']['r2'],
    'val_rmse': evals_result['val']['rmse'],
    'val_mae': evals_result['val']['mae'],
    'val_r2': evals_result['val']['r2'],
})
history_df.to_csv("training_metrics_per_iteration.csv", index=False)
print("✅ Saved per-iteration metrics to 'training_metrics_per_iteration.csv'")


# ====================== 4. EVALUATION ======================
def evaluate_model(model, X_scaled, y_true, set_name):
    preds = model.predict(xgb.DMatrix(X_scaled))
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)
    print(f"\n📊 {set_name} Performance:")
    print(f"MAE  : {mae:.3f} kWh")
    print(f"RMSE : {rmse:.3f} kWh")
    print(f"R²   : {r2:.4f}")
    return mae, rmse, r2

print("\n=== BASELINE XGBoost RESULTS (50:10:40 Split) ===")
train_mae, train_rmse, train_r2 = evaluate_model(model, X_train_scaled, y_train, "Train")
val_mae,   val_rmse,   val_r2   = evaluate_model(model, X_val_scaled,   y_val,   "Validation")
test_mae,  test_rmse,  test_r2  = evaluate_model(model, X_test_scaled,  y_test,  "Test (40%)")

# ====================== 5. LOG ======================
results_row = pd.DataFrame([{
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'experiment': 'Stratified_50_10_40',
    'n_train': len(X_train),
    'n_val': len(X_val),
    'n_test': len(X_test),
    'train_mae': train_mae,
    'train_rmse': train_rmse,
    'train_r2': train_r2,
    'val_mae': val_mae,
    'val_rmse': val_rmse,
    'val_r2': val_r2,
    'test_mae': test_mae,
    'test_rmse': test_rmse,
    'test_r2': test_r2
}])

log_file = "baseline_results_log.csv"
if os.path.exists(log_file):
    results_row.to_csv(log_file, mode='a', header=False, index=False)
else:
    results_row.to_csv(log_file, index=False)

print(f"\n📁 Results appended to {log_file}")

# ====================== 6. PRACTICAL INFERENCE ======================
print("\n🔍 Practical Inference Example:")
example = pd.DataFrame([{
    'parsed_milesRequested': 80.0, 'parsed_WhPerMile': 300.0,
    'parsed_minutesAvailable': 250.0, 'parsed_kWhRequested': 24.0,
    'revisionCount': 1, 'hour_of_day': 14,
    'day_of_week_encoded': day_le.transform(['Wednesday'])[0],
    'is_weekend': 0,
    'urgency_score': 65.0, 'flexibility_index': 0.72,
    'habit_stability': 0.85, 'grid_impact_proxy_encoded': 1,
    'urgency_flex_interaction': 65*0.72, 'revision_urgency': 65
}])

example_scaled = scaler.transform(example[feature_cols])
pred = model.predict(xgb.DMatrix(example_scaled))[0]
print(f"→ Predicted kWhDelivered: **{pred:.2f} kWh**")

# Save model
os.makedirs("xgboost_models_v2", exist_ok=True)
model.save_model("xgboost_models_v2/xgboost_baseline_50_10_40.json")
import joblib
joblib.dump(scaler, "xgboost_models_v2/scaler_baseline_50_10_40.pkl")
joblib.dump(day_le, "xgboost_models_v2/day_le.pkl")
joblib.dump(grid_le, "xgboost_models_v2/grid_le.pkl")

print("\n✅ XGBoost baseline (50:10:40) saved.")