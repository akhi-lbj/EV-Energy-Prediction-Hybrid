import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime
import os

# ====================== CONFIG ======================
MODEL_DIR = "ensemble_models_v16"      # for encoders
PROB_DIR = "prob_models"

# Features exactly as used in training
BASE_FEATURES = [
    'parsed_milesRequested', 'parsed_WhPerMile', 'parsed_minutesAvailable',
    'parsed_kWhRequested', 'revisionCount', 'hour_of_day', 'day_of_week_encoded',
    'is_weekend', 'urgency_score', 'flexibility_index', 'habit_stability',
    'grid_impact_proxy', 'request_efficiency', 'station_encoded', 'user_encoded'
]

BEHAVIORAL_FEATURES = ['urgency_flex_interaction', 'revision_urgency', 'habit_grid', 'energy_per_minute']

# ====================== LOAD MODELS ======================
print("🚀 Loading Probabilistic Models...")

# Target Encoders
te_station = joblib.load(f"{MODEL_DIR}/te_station.pkl")
te_user = joblib.load(f"{MODEL_DIR}/te_user.pkl")

# LightGBM Quantile models
lgb_models = {}
for alpha in [5, 50, 95]:
    model = lgb.Booster(model_file=f"{PROB_DIR}/lgb_quantile_alpha_{alpha}.txt")
    lgb_models[alpha] = model

# XGBoost Quantile models
xgb_models = {}
for alpha in [5, 50, 95]:
    model = xgb.Booster()
    model.load_model(f"{PROB_DIR}/xgb_quantile_alpha_{alpha}.model")
    xgb_models[alpha] = model

print("✅ All models loaded successfully!\n")

# ====================== HELPER FUNCTIONS ======================
def prepare_features(input_dict: dict) -> pd.DataFrame:
    """Convert raw input into exact feature vector used during training"""
    df = pd.DataFrame([input_dict])
    
    # Temporal features
    conn_time = pd.to_datetime(input_dict['connectionTime'])
    df['hour_of_day'] = conn_time.hour
    df['month'] = conn_time.month
    df['day_of_week_encoded'] = pd.factorize([input_dict.get('day_of_week', 'Monday')])[0][0]
    df['is_weekend'] = 1 if df['day_of_week_encoded'] in [5, 6] else 0   # rough approximation
    
    # Derived features
    df['energy_per_minute'] = df['parsed_kWhRequested'] / (df['parsed_minutesAvailable'] + 1e-5)
    df['request_efficiency'] = df['parsed_milesRequested'] / (df['parsed_WhPerMile'] + 1e-5)
    df['urgency_flex_interaction'] = df['urgency_score'] * df['flexibility_index']
    df['revision_urgency'] = df['revisionCount'] * df['urgency_score']
    df['habit_grid'] = df['habit_stability'] * df['grid_impact_proxy']
    
    # Target encoding
    df['station_encoded'] = te_station.transform(df[['stationID']]).flatten()
    df['user_encoded'] = te_user.transform(df[['parsed_userID']]).flatten()
    
    # Final feature order
    feat_cols = BASE_FEATURES + BEHAVIORAL_FEATURES
    return df[feat_cols]

def predict_probabilistic(sample_df: pd.DataFrame):
    """Return median + 90% PI from both LGB and XGB"""
    X = sample_df.values
    
    # LightGBM
    lgb_lower  = lgb_models[5].predict(X)
    lgb_median = lgb_models[50].predict(X)
    lgb_upper  = lgb_models[95].predict(X)
    
    # XGBoost
    dmatrix = xgb.DMatrix(X)
    xgb_lower  = xgb_models[5].predict(dmatrix)
    xgb_median = xgb_models[50].predict(dmatrix)
    xgb_upper  = xgb_models[95].predict(dmatrix)
    
    results = {
        "LightGBM": {
            "median_kWh": float(lgb_median[0]),
            "90_lower": float(lgb_lower[0]),
            "90_upper": float(lgb_upper[0]),
            "pi_width": float(lgb_upper[0] - lgb_lower[0])
        },
        "XGBoost": {
            "median_kWh": float(xgb_median[0]),
            "90_lower": float(xgb_lower[0]),
            "90_upper": float(xgb_upper[0]),
            "pi_width": float(xgb_upper[0] - xgb_lower[0])
        }
    }
    return results

# ====================== INTERACTIVE INFERENCE ======================
def main():
    print("=== EV Charging Energy Demand Probabilistic Forecaster ===\n")
    
    while True:
        print("Enter sample input (or type 'exit' to quit):")
        try:
            # Example values - you can change these
            sample = {
                "connectionTime": "2019-01-01 03:45:49",   # ← change this
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
            
            # Allow user to override
            print("\nDefault sample loaded. Press Enter to use it or provide new values (JSON style).")
            user_input = input("→ ").strip()
            if user_input.lower() == 'exit':
                break
            if user_input:
                import json
                sample.update(json.loads(user_input))
            
            input_df = prepare_features(sample)
            results = predict_probabilistic(input_df)
            
            print("\n" + "="*60)
            print(f"🔮 PREDICTION @ {sample['connectionTime']}")
            print("="*60)
            for model_name, pred in results.items():
                print(f"\n{model_name}:")
                print(f"   Median Energy Demand : {pred['median_kWh']:.3f} kWh")
                print(f"   90% Prediction Interval: [{pred['90_lower']:.3f}, {pred['90_upper']:.3f}] kWh")
                print(f"   Interval Width       : {pred['pi_width']:.3f} kWh")
            
            # Ensemble median (simple average)
            ensemble_median = (results["LightGBM"]["median_kWh"] + results["XGBoost"]["median_kWh"]) / 2
            print(f"\n📊 ENSEMBLE MEDIAN: {ensemble_median:.3f} kWh")
            print("="*60)
            
        except Exception as e:
            print(f"Error: {e}. Please check input format.")

if __name__ == "__main__":
    main()