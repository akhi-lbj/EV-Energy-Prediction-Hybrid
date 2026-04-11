import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import xgboost as xgb
import os
from datetime import datetime

print("🚀 EV Probabilistic Energy Demand Inference (Production Ready)")

# ====================== CONFIG ======================
PROB_DIR = "prob_models"          # Folder from your training script
ENCODER_DIR = "ensemble_models_v16"

# Exact features used in training
BASE_FEATURES = [
    'parsed_milesRequested', 'parsed_WhPerMile', 'parsed_minutesAvailable',
    'parsed_kWhRequested', 'revisionCount', 'hour_of_day', 'day_of_week_encoded',
    'is_weekend', 'urgency_score', 'flexibility_index', 'habit_stability',
    'grid_impact_proxy', 'request_efficiency', 'station_encoded', 'user_encoded'
]

BEHAVIORAL_FEATURES = ['urgency_flex_interaction', 'revision_urgency', 'habit_grid', 'energy_per_minute']

# ====================== LOAD MODELS & ENCODERS ======================
te_station = joblib.load(f"{ENCODER_DIR}/te_station.pkl")
te_user = joblib.load(f"{ENCODER_DIR}/te_user.pkl")

# LightGBM Quantile Models
lgb_models = {}
for alpha in [5, 50, 95]:
    txt_path = f"{PROB_DIR}/lgb_quantile_alpha_{alpha}.txt"
    pkl_path = f"{PROB_DIR}/lgb_quantile_alpha_{alpha}.pkl"   # fallback if you add it later
    
    if os.path.exists(pkl_path):
        lgb_models[alpha] = joblib.load(pkl_path)
    else:
        lgb_models[alpha] = lgb.Booster(model_file=txt_path)

# XGBoost Quantile Models
xgb_models = {}
for alpha in [5, 50, 95]:
    model = xgb.Booster()
    model.load_model(f"{PROB_DIR}/xgb_quantile_alpha_{alpha}.model")
    xgb_models[alpha] = model

print("✅ All probabilistic models loaded successfully!\n")

# ====================== FEATURE PREPARATION ======================
def prepare_features(sample: dict) -> pd.DataFrame:
    df = pd.DataFrame([sample])
    
    # Temporal
    conn_time = pd.to_datetime(sample['connectionTime'])
    df['hour_of_day'] = conn_time.hour
    df['day_of_week_encoded'] = pd.factorize([sample.get('day_of_week', 'Monday')])[0][0]
    df['is_weekend'] = 1 if df['day_of_week_encoded'] >= 5 else 0
    
    # Derived features
    df['energy_per_minute'] = df['parsed_kWhRequested'] / (df['parsed_minutesAvailable'] + 1e-5)
    df['request_efficiency'] = df['parsed_milesRequested'] / (df['parsed_WhPerMile'] + 1e-5)
    df['urgency_flex_interaction'] = df['urgency_score'] * df['flexibility_index']
    df['revision_urgency'] = df['revisionCount'] * df['urgency_score']
    df['habit_grid'] = df['habit_stability'] * df['grid_impact_proxy']
    
    # Target Encoding
    df['station_encoded'] = te_station.transform(df[['stationID']]).flatten()
    df['user_encoded'] = te_user.transform(df[['parsed_userID']]).flatten()
    
    feat_cols = BASE_FEATURES + BEHAVIORAL_FEATURES
    return df[feat_cols]

# ====================== PREDICTION ======================
def predict_probabilistic(sample_dict: dict):
    X_df = prepare_features(sample_dict)
    X = X_df.values
    
    # LightGBM Predictions
    lgb_lower  = lgb_models[5].predict(X)
    lgb_median = lgb_models[50].predict(X)
    lgb_upper  = lgb_models[95].predict(X)
    
    # XGBoost Predictions
    dtest = xgb.DMatrix(X)
    xgb_lower  = xgb_models[5].predict(dtest)
    xgb_median = xgb_models[50].predict(dtest)
    xgb_upper  = xgb_models[95].predict(dtest)
    
    results = {
        "LightGBM": {
            "median_kWh": round(float(lgb_median[0]), 3),
            "90%_PI": [round(float(lgb_lower[0]), 3), round(float(lgb_upper[0]), 3)],
            "width_kWh": round(float(lgb_upper[0] - lgb_lower[0]), 3)
        },
        "XGBoost": {
            "median_kWh": round(float(xgb_median[0]), 3),
            "90%_PI": [round(float(xgb_lower[0]), 3), round(float(xgb_upper[0]), 3)],
            "width_kWh": round(float(xgb_upper[0] - xgb_lower[0]), 3)
        }
    }
    
    # Simple ensemble median
    ensemble_median = round((results["LightGBM"]["median_kWh"] + results["XGBoost"]["median_kWh"]) / 2, 3)
    
    print("="*70)
    print(f"🔮 EV CHARGING DEMAND PROBABILISTIC FORECAST")
    print(f"Time: {sample_dict['connectionTime']}")
    print("="*70)
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"   Median Demand      : {res['median_kWh']} kWh")
        print(f"   90% Prediction Interval : {res['90%_PI']} kWh")
        print(f"   Interval Width     : {res['width_kWh']} kWh")
    print(f"\n📊 ENSEMBLE MEDIAN: {ensemble_median} kWh")
    print("="*70)
    
    return results, ensemble_median

# ====================== SAMPLE INPUTS ======================
if __name__ == "__main__":
    # Example 1: High demand session from your data
    sample1 = {
        "connectionTime": "2019-01-01 03:45:49",
        "stationID": "2-39-138-29",
        "parsed_userID": 489,
        "parsed_milesRequested": 210,
        "parsed_WhPerMile": 475,
        "parsed_minutesAvailable": 641,
        "parsed_kWhRequested": 99.75,
        "revisionCount": 2,
        "urgency_score": 0.92,
        "flexibility_index": 0.69,
        "habit_stability": 0.52,
        "grid_impact_proxy": 2,
        "day_of_week": "Tuesday"
    }
    
    # Example 2: Low demand session
    sample2 = {
        "connectionTime": "2019-12-30 19:02:40",
        "stationID": "2-39-130-31",
        "parsed_userID": 1920,
        "parsed_milesRequested": 20,
        "parsed_WhPerMile": 400,
        "parsed_minutesAvailable": 60,
        "parsed_kWhRequested": 8.0,
        "revisionCount": 1,
        "urgency_score": 0.9,
        "flexibility_index": 0.8,
        "habit_stability": 0.8,
        "grid_impact_proxy": 1,
        "day_of_week": "Monday"
    }
    
    print("Testing Sample 1 (High demand)...")
    predict_probabilistic(sample1)
    
    print("\nTesting Sample 2 (Low demand)...")
    predict_probabilistic(sample2)
    
    # Interactive mode
    print("\n💡 You can now call predict_probabilistic(your_dict) with custom values!")