import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import json
import warnings
warnings.filterwarnings('ignore')

print("Generating frontend data...")

# Load saved models
rf = joblib.load("sota_models_v3/rf_base.pkl")
xgb_model = joblib.load("sota_models_v3/xgb_base.pkl")
cat_model = joblib.load("sota_models_v3/cat_base.pkl")
lgb_base = joblib.load("sota_models_v3/lgb_base.pkl")
meta_model = lgb.Booster(model_file="sota_models_v3/meta_lgb.txt")

te_station = joblib.load("sota_models_v2/te_station.pkl")
te_user = joblib.load("sota_models_v2/te_user.pkl")

# Prepare dataset
df = pd.read_csv("acn_enhanced_final_2019_data.csv")
df['connectionTime'] = pd.to_datetime(df['connectionTime'], errors='coerce', utc=True)
df['disconnectTime'] = pd.to_datetime(df['disconnectTime'], errors='coerce', utc=True)

df['hour'] = df['connectionTime'].dt.hour
df['month'] = df['connectionTime'].dt.month
df['day_of_week_encoded'] = pd.factorize(df['day_of_week'].fillna('Monday'))[0]
df['is_weekend'] = (df['day_of_week_encoded'] >= 5).astype(int)
df['duration_hours'] = (df['disconnectTime'] - df['connectionTime']).dt.total_seconds() / 3600

from _internal_features import add_advanced_features
df = add_advanced_features(df)

# Derived features
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

# Just use the whole dataset for the frontend to have a large pool of random sessions
df_test = df.sample(n=min(500, len(df)), random_state=42).copy()

df_test['station_encoded'] = te_station.transform(df_test[['stationID']]).flatten()
df_test['user_encoded'] = te_user.transform(df_test[['parsed_userID']]).flatten()

X_test = df_test[feature_cols + ['station_encoded', 'user_encoded']].copy()

# Point forecast
stack_test = np.column_stack((
    rf.predict(X_test), 
    xgb_model.predict(X_test), 
    cat_model.predict(X_test), 
    lgb_base.predict(X_test)
))
point_pred = meta_model.predict(stack_test)

# Probabilistic
quantile_models = {}
for alpha in [0.05, 0.50, 0.95]:
    quantile_models[alpha] = lgb.Booster(model_file=f"sota_models_v2/lgb_quantile_alpha_{int(alpha*100)}.txt")

q05 = quantile_models[0.05].predict(X_test)
q50 = quantile_models[0.50].predict(X_test)
q95 = quantile_models[0.95].predict(X_test)

q_hat = 0.629
lower = q05 - q_hat
upper = q95 + q_hat

# Construct JSON output
frontend_data = []
for i in range(len(df_test)):
    row = df_test.iloc[i]
    frontend_data.append({
        'sessionID': str(row.get('sessionID', f'session_{i}')),
        'parsed_kWhRequested': float(row['parsed_kWhRequested']),
        'parsed_minutesAvailable': float(row['parsed_minutesAvailable']),
        'connectionTime': str(row['connectionTime']),
        'disconnectTime': str(row['disconnectTime']),
        'charging_efficiency': float(row['charging_efficiency']),
        'requested_gap': float(row['requested_gap']),
        'urgency_score': float(row['urgency_score']),
        'pred_median': float(q50[i]),
        'lower_conformal': float(lower[i]),
        'upper_conformal': float(upper[i]),
    })

output_path = 'app/app/frontend_data.json'
with open(output_path, 'w') as f:
    json.dump(frontend_data, f, indent=2)

print(f"✅ Generated {len(frontend_data)} sessions and saved to {output_path}")
