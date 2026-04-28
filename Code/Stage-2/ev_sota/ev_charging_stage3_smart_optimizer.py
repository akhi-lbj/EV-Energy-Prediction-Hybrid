import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import *
import joblib
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

print("Stage-3: Smart Priority-Aware Dynamic Optimization Module (Full Visuals)")

# ====================== 1. LOAD MODELS ======================
rf = joblib.load("sota_models_v3/rf_base.pkl")
xgb_m = joblib.load("sota_models_v3/xgb_base.pkl")
cat_m = joblib.load("sota_models_v3/cat_base.pkl")
lgb_base = joblib.load("sota_models_v3/lgb_base.pkl")
meta_model = lgb.Booster(model_file="sota_models_v3/meta_lgb.txt")

quantile_models = {}
for alpha in [0.05, 0.50, 0.95]:
    quantile_models[alpha] = lgb.Booster(model_file=f"sota_models_v2/lgb_quantile_alpha_{int(alpha*100)}.txt")

te_station = joblib.load("sota_models_v3/te_station.pkl")
te_user = joblib.load("sota_models_v3/te_user.pkl")

# ====================== 2. SELECT DAY ======================
df = pd.read_csv("acn_enhanced_final_2019_data.csv")
df['connectionTime'] = pd.to_datetime(df['connectionTime'], errors='coerce', utc=True)
df['disconnectTime'] = pd.to_datetime(df['disconnectTime'], errors='coerce', utc=True)

selected_date = '2019-07-26'   # ← Change this date as needed
df['date'] = df['connectionTime'].dt.date.astype(str)
df_day = df[df['date'] == selected_date].copy()

print(f"Selected Date: {selected_date} | Total sessions: {len(df_day)}")

# ====================== 3. FEATURE ENGINEERING ======================
df_day['hour'] = df_day['connectionTime'].dt.hour
df_day['day_of_week_encoded'] = pd.factorize(df_day['day_of_week'].fillna('Monday'))[0]
df_day['is_weekend'] = (df_day['day_of_week_encoded'] >= 5).astype(int)
df_day['duration_hours'] = (df_day['disconnectTime'] - df_day['connectionTime']).dt.total_seconds() / 3600
from _internal_features import add_advanced_features
df_day = add_advanced_features(df_day)
df_day['energy_per_minute'] = df_day['parsed_kWhRequested'] / (df_day['parsed_minutesAvailable'] + 1e-5)
df_day['request_efficiency'] = df_day['parsed_milesRequested'] / (df_day['parsed_WhPerMile'] + 1e-5)
df_day['urgency_flex_interaction'] = df_day['urgency_score'] * df_day['flexibility_index']
df_day['revision_urgency'] = df_day['revisionCount'] * df_day['urgency_score']
df_day['habit_grid'] = df_day['habit_stability'] * df_day['grid_impact_proxy'].fillna(1)
df_day['is_peak'] = df_day['hour'].between(17, 21).astype(int)
df_day['hour_sin'] = np.sin(2 * np.pi * df_day['hour'] / 24)
df_day['hour_cos'] = np.cos(2 * np.pi * df_day['hour'] / 24)

feature_cols = ['parsed_milesRequested', 'parsed_WhPerMile', 'parsed_minutesAvailable', 'parsed_kWhRequested',
                'revisionCount', 'hour', 'day_of_week_encoded', 'is_weekend', 'is_peak',
                'urgency_score', 'flexibility_index', 'habit_stability', 'grid_impact_proxy',
                'urgency_flex_interaction', 'revision_urgency', 'habit_grid', 'energy_per_minute',
                'request_efficiency', 'duration_hours', 'charging_efficiency', 'requested_gap',
                'hour_sin', 'hour_cos']

df_day = df_day.dropna(subset=feature_cols + ['parsed_userID', 'stationID']).copy()

df_day['station_encoded'] = te_station.transform(df_day[['stationID']]).flatten()
df_day['user_encoded'] = te_user.transform(df_day[['parsed_userID']]).flatten()
feature_cols += ['station_encoded', 'user_encoded']

X_day = df_day[feature_cols].copy()

# ====================== 4. PREDICTIONS ======================
stack_pred = np.column_stack((rf.predict(X_day), xgb_m.predict(X_day), cat_m.predict(X_day), lgb_base.predict(X_day)))
point_pred = meta_model.predict(stack_pred)

q05 = quantile_models[0.05].predict(X_day)
q50 = quantile_models[0.50].predict(X_day)
q95 = quantile_models[0.95].predict(X_day)

q_hat = 0.629
lower = q05 - q_hat
upper = q95 + q_hat

df_day['pred_kWh'] = point_pred
df_day['lower_kWh'] = lower
df_day['upper_kWh'] = upper
df_day['remaining_kWh'] = df_day['parsed_kWhRequested'] - df_day['kWhDelivered'].fillna(0)
df_day['minutes_left'] = df_day['parsed_minutesAvailable'].fillna(300)

active_sessions = df_day[df_day['remaining_kWh'] > 0.5].copy().reset_index(drop=True)
print(f"Optimizing for {len(active_sessions)} active sessions")

# ====================== 5. SMART OPTIMIZER ======================
def optimize_charging_schedule_smart(sessions_df, time_horizon_hours=8, time_step_min=15, grid_max_kw=400):
    """
    Smart Priority-Aware Optimizer - Fixed Infeasible Issue
    """
    prob = LpProblem("Smart_EV_Charging_Optimization", LpMinimize)
    
    time_steps = int(time_horizon_hours * 60 / time_step_min)
    slots = list(range(time_steps))
    
    power = LpVariable.dicts("power", ((i, t) for i in sessions_df.index for t in slots), 
                             lowBound=0, cat='Continuous')
    peak_power = LpVariable("peak_power", lowBound=0)
    
    urgency_weight = sessions_df['urgency_flex_interaction'].fillna(1.0)
    
    # Objective: Minimize peak load + penalize delaying urgent users
    prob += (peak_power + 
             0.01 * lpSum(power[i,t] for i in sessions_df.index for t in slots) +
             0.06 * lpSum(urgency_weight[i] * power[i,t] * (time_steps - t) 
                         for i in sessions_df.index for t in slots))
    
    charger_max = 11.0
    
    for i in sessions_df.index:
        sess = sessions_df.loc[i]
        required = max(sess.get('remaining_kWh', 0), sess.get('lower_kWh', 0))
        
        max_slots = int(sess.get('minutes_left', 300) / time_step_min)
        
        # Energy delivery requirement (Smart cap)
        # We cannot ask the charger for more energy than it can physically deliver in the time remaining
        physical_max_kwh = max_slots * charger_max * (time_step_min / 60.0)
        achievable_target = min(required * 0.93, physical_max_kwh)
        
        prob += lpSum(power[i,t] * (time_step_min / 60.0) for t in slots) >= achievable_target
        
        # Charger hardware limit
        for t in slots:
            prob += power[i,t] <= charger_max
        
        # Only apply mild restriction for very urgent users
        if sess.get('urgency_score', 50) > 80 and sess.get('flexibility_index', 0.5) < 0.3:
            # Very urgent + very inflexible → encourage finishing earlier, but not strictly
            soft_limit = int(max_slots * 0.9)
            for t in range(soft_limit, time_steps):
                prob += power[i,t] <= 2.0          # allow small trickle charging
        else:
            # Normal users - very soft constraint
            for t in range(max_slots, time_steps):
                prob += power[i,t] <= 0.5          # almost zero after deadline
    
    # Grid-wide power limit
    for t in slots:
        prob += lpSum(power[i,t] for i in sessions_df.index) <= grid_max_kw
        prob += lpSum(power[i,t] for i in sessions_df.index) <= peak_power
    
    # Solve
    status = prob.solve(PULP_CBC_CMD(msg=0, timeLimit=90))
    print(f"Status: {LpStatus[status]} | Optimal Peak: {value(peak_power):.2f} kW")
    
    # Extract schedule
    schedule = []
    for i in sessions_df.index:
        for t in slots:
            p = value(power[i, t])
            if p > 0.1:
                schedule.append({
                    'session_id': i,
                    'user_id': int(sessions_df.loc[i, 'parsed_userID']),
                    'time_slot': t,
                    'power_kw': round(p, 2),
                    'energy_kwh': round(p * time_step_min/60, 3),
                    'urgency': round(sessions_df.loc[i].get('urgency_score', 0), 1)
                })
    
    return pd.DataFrame(schedule), value(peak_power)

# Run Optimization
schedule_df, optimal_peak = optimize_charging_schedule_smart(active_sessions, 
                                                             time_horizon_hours=8, 
                                                             time_step_min=15, 
                                                             grid_max_kw=350)

# ====================== 6. VISUALIZATIONS ======================

# 6.1 Before vs After Aggregate Power + Peak Bar
def plot_before_after(active_sessions, schedule_df, optimal_peak, selected_date):
    fig = plt.figure(figsize=(15, 10))
    
    # --- Aggregate Power Line Chart ---
    ax1 = plt.subplot(2, 1, 1)
    n_slots = 40
    time_index = [f"T{t}" for t in range(n_slots)]
    
    # Naive Power
    naive_power = np.zeros(n_slots)
    charger_max = 11.0
    for _, row in active_sessions.iterrows():
        remaining = row.get('remaining_kWh', 10)
        slots_needed = int(np.ceil(remaining / (charger_max * 15 / 60))) + 2
        for t in range(min(slots_needed, n_slots)):
            naive_power[t] += charger_max
    
    # Optimized Power
    optimized_power = np.zeros(n_slots)
    if not schedule_df.empty:
        for _, row in schedule_df.iterrows():
            ts = int(row['time_slot'])
            if ts < n_slots:
                optimized_power[ts] += row['power_kw']
    
    ax1.plot(time_index, naive_power, label='Naive (Full Power)', color='crimson', linewidth=2.5)
    ax1.plot(time_index, optimized_power, label='Optimized Schedule', color='forestgreen', linewidth=2.5)
    ax1.set_ylabel('Total Power Demand (kW)')
    ax1.set_title(f'Before vs After Optimization - {selected_date}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Peak Reduction Bar ---
    ax2 = plt.subplot(2, 1, 2)
    naive_peak = naive_power.max()
    opt_peak = optimized_power.max()
    reduction = ((naive_peak - opt_peak) / naive_peak * 100) if naive_peak > 0 else 0
    
    bars = ax2.bar(['Naive Peak', 'Optimized Peak'], [naive_peak, opt_peak], color=['crimson', 'forestgreen'])
    ax2.set_ylabel('Peak Power (kW)')
    ax2.set_title(f'Peak Load Reduction: {reduction:.1f}%')
    
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 8, f'{h:.1f} kW', ha='center')
    
    plt.tight_layout()
    plt.savefig(f'before_after_optimization_{selected_date}.png', dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Before vs After chart saved: before_after_optimization_{selected_date}.png")

# 6.2 Colored Gantt Chart (Urgency Aware)
def plot_gantt_chart(schedule_df, active_sessions, selected_date, optimal_peak):
    if schedule_df.empty:
        print("No schedule generated.")
        return
    
    top_users = active_sessions['parsed_userID'].value_counts().head(15).index.tolist()
    user_map = {uid: idx for idx, uid in enumerate(top_users)}
    
    plt.figure(figsize=(16, 11))
    
    for _, row in schedule_df.iterrows():
        uid = row['user_id']
        if uid not in user_map: continue
        y = user_map[uid]
        start = row['time_slot']
        urgency = row.get('urgency', 50)
        
        if urgency >= 75:
            color = '#d32f2f'   # Dark Red
        elif urgency >= 60:
            color = '#ff9800'   # Orange
        elif urgency >= 45:
            color = '#ffeb3b'   # Yellow
        else:
            color = '#4caf50'   # Green
            
        plt.barh(y, 1, left=start, height=0.65, color=color, alpha=0.9, edgecolor='black')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d32f2f', label='Very High Urgency (≥75)'),
        Patch(facecolor='#ff9800', label='High Urgency (60-74)'),
        Patch(facecolor='#ffeb3b', label='Medium Urgency (45-59)'),
        Patch(facecolor='#4caf50', label='Low Urgency (<45)')
    ]
    plt.legend(handles=legend_elements, title='Urgency Level', loc='upper right')
    
    plt.yticks(range(len(top_users)), [f"User {int(uid)}" for uid in top_users])
    plt.xlabel('Time Slot (15-minute intervals)')
    plt.ylabel('User ID (Different Cars)')
    plt.title(f'Smart EV Charging Gantt Chart - {selected_date}\nPeak Load: {optimal_peak:.1f} kW (Priority + Urgency Aware)')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'gantt_chart_urgency_{selected_date}.png', dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Urgency-colored Gantt Chart saved: gantt_chart_urgency_{selected_date}.png")

# ====================== RUN EVERYTHING ======================
plot_before_after(active_sessions, schedule_df, optimal_peak, selected_date)
plot_gantt_chart(schedule_df, active_sessions, selected_date, optimal_peak)

print("\nStage-3 Complete!")
print(f"Date: {selected_date} | Sessions: {len(active_sessions)} | Final Peak: {optimal_peak:.1f} kW")
print("Both charts generated successfully!")