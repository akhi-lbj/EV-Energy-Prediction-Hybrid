import sys
import argparse
import json
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['fetch', 'predict'])
parser.add_argument('--idx', type=int, default=None)
args = parser.parse_args()

# Load quickly
df = pd.read_csv("acn_enhanced_final_2019_data.csv")
df = df.dropna(subset=['kWhDelivered', 'parsed_kWhRequested', 'parsed_minutesAvailable']).reset_index(drop=True)

if args.mode == 'fetch':
    # Pick a random row
    random_idx = int(np.random.randint(0, len(df)))
    row = df.iloc[random_idx]
    
    output = {
        'row_idx': random_idx,
        'parsed_milesRequested': float(row.get('parsed_milesRequested', 0)),
        'parsed_WhPerMile': float(row.get('parsed_WhPerMile', 0)),
        'parsed_minutesAvailable': float(row.get('parsed_minutesAvailable', 0)),
        'parsed_kWhRequested': float(row.get('parsed_kWhRequested', 0)),
        'revisionCount': float(row.get('revisionCount', 0)),
        'connectionTime': str(row.get('connectionTime', '')),
        'disconnectTime': str(row.get('disconnectTime', '')),
        'day_of_week': str(row.get('day_of_week', '')),
        'urgency_score': float(row.get('urgency_score', 50.0)),
        'flexibility_index': float(row.get('flexibility_index', 1.0)),
        'habit_stability': float(row.get('habit_stability', 1.0)),
        'grid_impact_proxy': float(row.get('grid_impact_proxy', 1.0)),
        'true_kWhDelivered': float(row.get('kWhDelivered', 0))
    }
    print(json.dumps(output))
    sys.exit(0)

elif args.mode == 'predict':
    if args.idx is None or args.idx < 0 or args.idx >= len(df):
        print(json.dumps({"error": "Invalid row index"}))
        sys.exit(1)
        
    row = df.iloc[args.idx].copy()
    row_df = pd.DataFrame([row])

    # Import ML libraries only when predicting to keep fetch fast
    import joblib
    import lightgbm as lgb
    te_station = joblib.load("sota_models_v2/te_station.pkl")
    te_user = joblib.load("sota_models_v2/te_user.pkl")
    
    # Compute ALL features on this row
    row_df['connectionTime'] = pd.to_datetime(row_df['connectionTime'], errors='coerce', utc=True)
    row_df['disconnectTime'] = pd.to_datetime(row_df['disconnectTime'], errors='coerce', utc=True)

    row_df['hour'] = row_df['connectionTime'].dt.hour
    row_df['month'] = row_df['connectionTime'].dt.month
    row_df['day_of_week_encoded'] = pd.factorize(row_df['day_of_week'].fillna('Monday'))[0]
    row_df['is_weekend'] = (row_df['day_of_week_encoded'] >= 5).astype(int)

    from _internal_features import add_advanced_features
    row_df = add_advanced_features(row_df)

    row_df['energy_per_minute'] = row_df['parsed_kWhRequested'] / (row_df['parsed_minutesAvailable'] + 1e-5)
    row_df['request_efficiency'] = row_df['parsed_milesRequested'] / (row_df['parsed_WhPerMile'] + 1e-5)

    # Defaults
    if 'urgency_score' not in row_df.columns: row_df['urgency_score'] = 50.0
    if 'flexibility_index' not in row_df.columns: row_df['flexibility_index'] = 1.0
    if 'grid_impact_proxy' not in row_df.columns: row_df['grid_impact_proxy'] = 1.0
    if 'habit_stability' not in row_df.columns: row_df['habit_stability'] = 1.0

    row_df['urgency_flex_interaction'] = row_df['urgency_score'] * row_df['flexibility_index']
    row_df['revision_urgency'] = row_df.get('revisionCount', 0) * row_df['urgency_score']
    row_df['habit_grid'] = row_df['habit_stability'] * row_df['grid_impact_proxy'].fillna(1)
    row_df['is_peak'] = row_df['hour'].between(17, 21).astype(int)
    row_df['hour_sin'] = np.sin(2 * np.pi * row_df['hour'] / 24)
    row_df['hour_cos'] = np.cos(2 * np.pi * row_df['hour'] / 24)

    feature_cols = ['parsed_milesRequested', 'parsed_WhPerMile', 'parsed_minutesAvailable', 'parsed_kWhRequested',
                    'revisionCount', 'hour', 'day_of_week_encoded', 'is_weekend', 'is_peak',
                    'urgency_score', 'flexibility_index', 'habit_stability', 'grid_impact_proxy',
                    'urgency_flex_interaction', 'revision_urgency', 'habit_grid', 'energy_per_minute',
                    'request_efficiency', 'duration_hours', 'charging_efficiency', 'requested_gap',
                    'hour_sin', 'hour_cos']

    row_df['station_encoded'] = te_station.transform(row_df[['stationID']]).flatten()
    row_df['user_encoded'] = te_user.transform(row_df[['parsed_userID']]).flatten()

    for col in feature_cols:
        if col not in row_df.columns: row_df[col] = 0.0
    row_df = row_df.fillna(0)

    X_test = row_df[feature_cols + ['station_encoded', 'user_encoded']].copy()

    rf = joblib.load("sota_models_v3/rf_base.pkl")
    xgb_model = joblib.load("sota_models_v3/xgb_base.pkl")
    cat_model = joblib.load("sota_models_v3/cat_base.pkl")
    lgb_base = joblib.load("sota_models_v3/lgb_base.pkl")
    meta_model = lgb.Booster(model_file="sota_models_v3/meta_lgb.txt")

    stack_test = np.column_stack((
        rf.predict(X_test), 
        xgb_model.predict(X_test), 
        cat_model.predict(X_test), 
        lgb_base.predict(X_test)
    ))
    point_pred = meta_model.predict(stack_test)[0]

    q05 = lgb.Booster(model_file="sota_models_v2/lgb_quantile_alpha_5.txt").predict(X_test)[0]
    q50 = lgb.Booster(model_file="sota_models_v2/lgb_quantile_alpha_50.txt").predict(X_test)[0]
    q95 = lgb.Booster(model_file="sota_models_v2/lgb_quantile_alpha_95.txt").predict(X_test)[0]

    q_hat = 0.629
    lower = q05 - q_hat
    upper = q95 + q_hat

    output = {
        'pred_median': float(q50),
        'lower_conformal': float(lower),
        'upper_conformal': float(upper),
    }

    print(json.dumps(output))
