import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ====================== 1. LOAD & PREPROCESS ======================
print("🚀 Loading enhanced ACN 2019 dataset...")
df = pd.read_csv("acn_enhanced_final_2019_data.csv")

# Convert connectionTime to datetime and extract temporal features
df['connectionTime'] = pd.to_datetime(df['connectionTime'], errors='coerce', utc=True)
df['hour_of_day'] = df['connectionTime'].dt.hour
df['month'] = df['connectionTime'].dt.month

# Encode grid_impact_proxy
le = LabelEncoder()
df['grid_impact_proxy_encoded'] = le.fit_transform(df['grid_impact_proxy'].fillna('medium'))

# Create interaction features
df['urgency_flex_interaction'] = df['urgency_score'] * df['flexibility_index']
df['revision_urgency'] = df['revisionCount'] * df['urgency_score']

# Final feature list
feature_cols = [
    'parsed_milesRequested', 'parsed_WhPerMile', 'parsed_minutesAvailable',
    'parsed_kWhRequested', 'revisionCount', 'hour_of_day', 'day_of_week',
    'is_weekend', 'urgency_score', 'flexibility_index', 'habit_stability',
    'grid_impact_proxy_encoded', 'urgency_flex_interaction', 'revision_urgency'
]

target = 'kWhDelivered'

# Drop rows with missing target or key features
df = df.dropna(subset=[target] + feature_cols)

X = df[feature_cols]
y = df[target]

print(f"Dataset shape: {df.shape}")
print(f"Features: {len(feature_cols)}")

# ====================== 2. TRAIN/VAL/TEST SPLIT ======================
# Temporal split - full year 2019
train_mask = df['month'] <= 10
val_mask = df['month'] == 11
test_mask = df['month'] == 12

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# Scale features (XGBoost benefits from it in practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert to DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dval = xgb.DMatrix(X_val_scaled, label=y_val)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# ====================== 3. TRAIN BASELINE XGBoost ======================
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 1000,
    'random_state': 42,
    'tree_method': 'hist'
}

print("Training XGBoost Baseline...")
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=50,
    verbose_eval=100
)

# ====================== 4. EVALUATION ======================
def evaluate_model(model, X, y_true, set_name="Test"):
    preds = model.predict(xgb.DMatrix(X))
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)
    
    print(f"\n📊 {set_name} Performance:")
    print(f"MAE:  {mae:.3f} kWh")
    print(f"RMSE: {rmse:.3f} kWh")
    print(f"R²:   {r2:.4f}")
    return preds

print("\n=== BASELINE RESULTS ===")
_ = evaluate_model(model, X_train_scaled, y_train, "Train")
_ = evaluate_model(model, X_val_scaled, y_val, "Validation")
test_preds = evaluate_model(model, X_test_scaled, y_test, "Test (Dec 2019)")

# ====================== 5. INFERENCE EXAMPLE (Practical Usage) ======================
print("\n🔍 Example Inference (Practical Inputs):")
example_input = pd.DataFrame([{
    'parsed_milesRequested': 80,
    'parsed_WhPerMile': 300,
    'parsed_minutesAvailable': 250,
    'parsed_kWhRequested': 24,
    'revisionCount': 1,
    'hour_of_day': 14,
    'day_of_week': 3,
    'is_weekend': 0,
    'urgency_score': 65,           # learned pattern
    'flexibility_index': 0.72,
    'habit_stability': 0.85,
    'grid_impact_proxy_encoded': 1,  # medium
    'urgency_flex_interaction': 65 * 0.72,
    'revision_urgency': 1 * 65
}])

example_scaled = scaler.transform(example_input[feature_cols])
pred_kwh = model.predict(xgb.DMatrix(example_scaled))[0]

print(f"Predicted kWhDelivered: {pred_kwh:.2f} kWh")

# Save model and scaler
os.makedirs("models", exist_ok=True)
model.save_model("models/xgboost_baseline.json")
import joblib
joblib.dump(scaler, "models/scaler_baseline.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("\n✅ Baseline model saved to 'models/' folder")