import pandas as pd
import numpy as np
from pulp import *
import joblib
import lightgbm as lgb
from datetime import timedelta
import warnings
from sklearn.preprocessing import TargetEncoder
warnings.filterwarnings('ignore')

print("Stage-3: Dynamic Optimization Module - Grid-Aware EV Charging Scheduler")

# ====================== 1. LOAD PREDICTION MODELS ======================
# Load your best models from sota_models_v3 and v2
rf = joblib.load("sota_models_v3/rf_base.pkl")
xgb_m = joblib.load("sota_models_v3/xgb_base.pkl")
cat_m = joblib.load("sota_models_v3/cat_base.pkl")
lgb_base = joblib.load("sota_models_v3/lgb_base.pkl")
meta_model = lgb.Booster(model_file="sota_models_v3/meta_lgb.txt")

# Quantile models for uncertainty
quantile_models = {}
for alpha in [0.05, 0.50, 0.95]:
    quantile_models[alpha] = lgb.Booster(model_file=f"sota_models_v2/lgb_quantile_alpha_{int(alpha*100)}.txt")

te_station = joblib.load("sota_models_v3/te_station.pkl")
te_user = joblib.load("sota_models_v3/te_user.pkl")

# ====================== 2. LOAD & PREPARE LIVE SESSION DATA ======================
# In production, this would come from real-time stream (Kafka/MQTT)
df_live = pd.read_csv("acn_enhanced_final_2019_data.csv").sample(50, random_state=42)  # Simulate active sessions

# ====================== 1. DATA (same as Phase 3) ======================
df = pd.read_csv("acn_enhanced_final_2019_data.csv")
df['connectionTime'] = pd.to_datetime(df['connectionTime'], errors='coerce', utc=True)
df['disconnectTime'] = pd.to_datetime(df['disconnectTime'], errors='coerce', utc=True)

# Features (same robust set)
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

feature_cols = ['parsed_milesRequested', 'parsed_WhPerMile', 'parsed_minutesAvailable', 'parsed_kWhRequested',
                'revisionCount', 'hour', 'day_of_week_encoded', 'is_weekend', 'is_peak',
                'urgency_score', 'flexibility_index', 'habit_stability', 'grid_impact_proxy',
                'urgency_flex_interaction', 'revision_urgency', 'habit_grid', 'energy_per_minute',
                'request_efficiency', 'duration_hours', 'charging_efficiency', 'requested_gap',
                'hour_sin', 'hour_cos']

target = 'kWhDelivered'
df = df.dropna(subset=[target] + feature_cols).copy()

# 50:10:40 split
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

# Target Encoding
te_station = TargetEncoder(target_type='continuous', random_state=42)
te_user = TargetEncoder(target_type='continuous', random_state=42)

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

# Prepare X_live with same feature_cols
df_live['connectionTime'] = pd.to_datetime(df_live['connectionTime'], errors='coerce', utc=True)
df_live['disconnectTime'] = pd.to_datetime(df_live['disconnectTime'], errors='coerce', utc=True)

df_live['hour'] = df_live['connectionTime'].dt.hour
df_live['month'] = df_live['connectionTime'].dt.month
df_live['day_of_week_encoded'] = pd.factorize(df_live['day_of_week'].fillna('Monday'))[0]
df_live['is_weekend'] = (df_live['day_of_week_encoded'] >= 5).astype(int)
df_live['duration_hours'] = (df_live['disconnectTime'] - df_live['connectionTime']).dt.total_seconds() / 3600
df_live['charging_efficiency'] = df_live['kWhDelivered'] / df_live['parsed_kWhRequested'].replace(0, np.nan)
df_live['requested_gap'] = df_live['parsed_kWhRequested'] - df_live['kWhDelivered']
df_live['energy_per_minute'] = df_live['parsed_kWhRequested'] / (df_live['parsed_minutesAvailable'] + 1e-5)
df_live['request_efficiency'] = df_live['parsed_milesRequested'] / (df_live['parsed_WhPerMile'] + 1e-5)
df_live['urgency_flex_interaction'] = df_live['urgency_score'] * df_live['flexibility_index']
df_live['revision_urgency'] = df_live['revisionCount'] * df_live['urgency_score']
df_live['habit_grid'] = df_live['habit_stability'] * df_live['grid_impact_proxy'].fillna(1)
df_live['is_peak'] = df_live['hour'].between(17, 21).astype(int)
df_live['hour_sin'] = np.sin(2 * np.pi * df_live['hour'] / 24)
df_live['hour_cos'] = np.cos(2 * np.pi * df_live['hour'] / 24)

df_live['station_encoded'] = te_station.transform(df_live[['stationID']]).flatten()
df_live['user_encoded'] = te_user.transform(df_live[['parsed_userID']]).flatten()

df_live = df_live.dropna(subset=feature_cols).copy()
X_live = df_live[feature_cols].copy()

# Predict point + uncertainty
stack_live = np.column_stack((
    rf.predict(X_live), xgb_m.predict(X_live), 
    cat_m.predict(X_live), lgb_base.predict(X_live)
))
point_pred = meta_model.predict(stack_live)

q05 = quantile_models[0.05].predict(X_live)
q50 = quantile_models[0.50].predict(X_live)
q95 = quantile_models[0.95].predict(X_live)

# Conformal adjustment (from your Phase 3)
q_hat = 0.629
lower = q05 - q_hat
upper = q95 + q_hat

# Add to dataframe
df_live['pred_kWh'] = point_pred
df_live['lower_kWh'] = lower
df_live['upper_kWh'] = upper
df_live['remaining_kWh'] = df_live['parsed_kWhRequested'] - df_live['kWhDelivered'].fillna(0)
df_live['minutes_left'] = df_live['parsed_minutesAvailable']

# Filter only active sessions needing charge
active_sessions = df_live[df_live['remaining_kWh'] > 0.5].copy()

print(f"Optimizing for {len(active_sessions)} active sessions")

# ====================== 3. DYNAMIC OPTIMIZATION (MILP with PuLP) ======================
def optimize_charging_schedule(sessions_df, time_horizon_hours=4, time_step_min=15, grid_max_kw=500):
    """
    Solves a MILP to schedule charging power for active sessions.
    """
    prob = LpProblem("EV_Charging_Optimization", LpMinimize)
    
    # Time slots
    time_steps = int(time_horizon_hours * 60 / time_step_min)
    slots = list(range(time_steps))
    
    # Decision variables: power (kW) for each session at each time slot
    power = LpVariable.dicts("power", 
                             ((i, t) for i in sessions_df.index for t in slots),
                             lowBound=0, cat='Continuous')
    
    # Auxiliary: peak power
    peak_power = LpVariable("peak_power", lowBound=0)
    
    # Objective: Minimize peak power (can add TOU cost later)
    prob += peak_power + 0.01 * lpSum(power[i,t] for i in sessions_df.index for t in slots)
    
    # Constraints
    for i in sessions_df.index:
        sess = sessions_df.loc[i]
        
        # 1. Energy requirement (conservative - use lower bound)
        required_energy = max(sess['remaining_kWh'], sess['lower_kWh'])
        prob += lpSum(power[i,t] * (time_step_min/60) for t in slots) >= required_energy * 0.95  # 95% of required
        
        # 2. Charger limit (assume 7-22 kW typical)
        charger_max = 11.0  # kW, adjust per station
        for t in slots:
            prob += power[i,t] <= charger_max
            
        # 3. Must finish before departure (approximate)
        max_slots = int(sess['minutes_left'] / time_step_min)
        for t in range(max_slots, time_steps):
            prob += power[i,t] <= 0.01  # almost zero after deadline
    
    # 4. Grid / station aggregate limit per time slot
    for t in slots:
        prob += lpSum(power[i,t] for i in sessions_df.index) <= grid_max_kw
        prob += lpSum(power[i,t] for i in sessions_df.index) <= peak_power   # track peak
    
    # Solve
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=60))
    
    print(f"Status: {LpStatus[prob.status]} | Optimal Peak: {value(peak_power):.2f} kW")
    
    # Extract schedule
    schedule = []
    for i in sessions_df.index:
        for t in slots:
            p = value(power[i,t])
            if p > 0.1:
                schedule.append({
                    'session_id': i,
                    'time_slot': t,
                    'power_kw': round(p, 2),
                    'energy_kwh': round(p * time_step_min/60, 3)
                })
    
    schedule_df = pd.DataFrame(schedule)
    return schedule_df, value(peak_power)

# ====================== 4. RUN OPTIMIZATION ======================
schedule_df, optimal_peak = optimize_charging_schedule(active_sessions, 
                                                       time_horizon_hours=6, 
                                                       time_step_min=15, 
                                                       grid_max_kw=300)

print("\nOptimized Charging Schedule Sample:")
print(schedule_df.head(20))

# Save
schedule_df.to_csv("optimized_charging_schedule.csv", index=False)
print(f"Optimal peak load reduced to {optimal_peak:.2f} kW")

# ====================== 5. EVALUATION & VISUALIZATION ======================
# Compare naive (full power) vs optimized peak
naive_peak = active_sessions['remaining_kWh'].sum() / 1.0  # rough
print(f"Naive peak estimate: ~{naive_peak:.1f} kW -> Optimized: {optimal_peak:.2f} kW")
print(f"Peak reduction: {((naive_peak - optimal_peak)/naive_peak*100):.1f}%")

print("\nStage-3 Dynamic Optimization Module COMPLETE!")
print("Ready for real-time integration with charging station controllers.")