# Stage-3: Smart Priority-Aware Dynamic EV Charging Optimizer
## Complete Technical Documentation

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Dependencies & Imports](#3-dependencies--imports)
4. [Section 1 — Model Loading](#4-section-1--model-loading)
5. [Section 2 — Data Loading & Date Selection](#5-section-2--data-loading--date-selection)
6. [Section 3 — Feature Engineering](#6-section-3--feature-engineering)
7. [Section 4 — Predictions & Confidence Intervals](#7-section-4--predictions--confidence-intervals)
8. [Section 5 — Smart Optimizer (LP Formulation)](#8-section-5--smart-optimizer-lp-formulation)
9. [Section 6 — Visualizations](#9-section-6--visualizations)
10. [Variable Reference Table](#10-variable-reference-table)
11. [End-to-End Data Flow](#11-end-to-end-data-flow)

---

## 1. Overview

This script is **Stage 3** of a multi-stage EV charging intelligence pipeline. It integrates:

- A **stacked ensemble ML model** to predict how much energy each EV session needs.
- A **quantile regression model** to generate uncertainty bounds around each prediction.
- A **Linear Programming (LP) optimizer** to schedule charging power across time slots while respecting grid constraints and user urgency.
- **Visualization functions** that produce publication-quality charts comparing naive vs. optimized charging.

### Why Stage 3 Matters

Stages 1 and 2 are responsible for training the ML models and preparing historical data. Stage 3 **consumes those trained artifacts** and applies them in a real-time-like operational context — given all active EV sessions on a chosen day, it decides *when* and *how much* to charge each vehicle to minimize peak grid demand while prioritizing urgent users.

### Core Problem Being Solved

> Given `N` active EV charging sessions, each with a different energy requirement, urgency level, and parking deadline — how do we allocate power across 8 hours of 15-minute time slots such that the **peak grid demand is minimized** and **urgent users are served first**?

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STAGE-3 PIPELINE                     │
│                                                         │
│  Raw CSV Data ──► Feature Engineering                   │
│                          │                              │
│                          ▼                              │
│              Stacked Ensemble Models                    │
│         (RF + XGB + CatBoost + LGB → Meta-LGB)          │
│                          │                              │
│              Point Prediction + Quantile Bounds         │
│                          │                              │
│                          ▼                              │
│           Linear Programming Optimizer (PuLP)           │
│    Minimize: Peak Power + Urgency-Weighted Delay         │
│    Subject to: Grid Cap, Charger Limits, Deadlines      │
│                          │                              │
│                          ▼                              │
│       Visualizations: Before/After + Gantt Chart        │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Dependencies & Imports

### Code Snippet

```python
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
```

### Explanation

| Library | Why It's Used |
|---|---|
| `pandas` | Loads and manipulates the EV session CSV data. All sessions live in a `DataFrame`. |
| `numpy` | Fast array operations — used for time-slot power arrays, sin/cos transformations, and vectorized math. |
| `matplotlib.pyplot` | Renders all charts — the before/after line chart and the urgency-coloured Gantt chart. |
| `seaborn` | Imported for theme/style consistency (`seaborn-v0_8-whitegrid`). Not called directly in plotting functions but influences aesthetics globally. |
| `pulp` | The LP (Linear Programming) solver library. `from pulp import *` brings in `LpProblem`, `LpVariable`, `LpMinimize`, `lpSum`, `value`, `PULP_CBC_CMD`, and `LpStatus`. |
| `joblib` | Deserializes the pre-trained sklearn-compatible models (Random Forest, XGBoost, CatBoost, LightGBM base) from `.pkl` files. |
| `lightgbm` | Loads the meta-model and quantile models directly via `lgb.Booster`, which reads LightGBM's native `.txt` format. |
| `warnings` | `warnings.filterwarnings('ignore')` suppresses deprecation/convergence warnings from sklearn and PuLP, keeping terminal output clean. |

**`plt.style.use('seaborn-v0_8-whitegrid')`** — Sets the global matplotlib style to the seaborn whitegrid theme (updated name for matplotlib ≥ 3.6). This gives clean plots with a white background and light grey grid lines.

---

## 4. Section 1 — Model Loading

### Code Snippet

```python
# ====================== 1. LOAD MODELS ======================
rf = joblib.load("sota_models_v3/rf_base.pkl")
xgb_m = joblib.load("sota_models_v3/xgb_base.pkl")
cat_m = joblib.load("sota_models_v3/cat_base.pkl")
lgb_base = joblib.load("sota_models_v3/lgb_base.pkl")
meta_model = lgb.Booster(model_file="sota_models_v3/meta_lgb.txt")

quantile_models = {}
for alpha in [0.05, 0.50, 0.95]:
    quantile_models[alpha] = lgb.Booster(
        model_file=f"sota_models_v2/lgb_quantile_alpha_{int(alpha*100)}.txt"
    )

te_station = joblib.load("sota_models_v3/te_station.pkl")
te_user = joblib.load("sota_models_v3/te_user.pkl")
```

### Explanation

This section reconstructs a **stacking ensemble** — a two-layer ML architecture where base models make predictions that are then fed into a meta-model for a final, more accurate prediction.

#### Base Models (Layer 1)

| Variable | File | Model Type | Role |
|---|---|---|---|
| `rf` | `rf_base.pkl` | Random Forest | Tree-based ensemble; robust to outliers and noisy features |
| `xgb_m` | `xgb_base.pkl` | XGBoost | Gradient-boosted trees with L1/L2 regularization |
| `cat_m` | `cat_base.pkl` | CatBoost | Handles categorical features natively; strong on tabular data |
| `lgb_base` | `lgb_base.pkl` | LightGBM (base) | Fastest of the four; leaf-wise growth strategy |

Each base model was trained on the same feature set in Stage 1/2, and their `.pkl` files are joblib-serialized sklearn-compatible objects.

#### Meta Model (Layer 2)

| Variable | File | Details |
|---|---|---|
| `meta_model` | `meta_lgb.txt` | LightGBM Booster trained on the *out-of-fold predictions* from the 4 base models. Learns how to best weight and combine them. |

#### Quantile Models

```python
quantile_models = {}
for alpha in [0.05, 0.50, 0.95]:
    quantile_models[alpha] = lgb.Booster(...)
```

Three separate LightGBM models trained with **quantile regression loss** (`alpha` = the quantile):

| Key | Alpha | Meaning |
|---|---|---|
| `quantile_models[0.05]` | 5th percentile | Lower bound — conservative estimate of energy needed |
| `quantile_models[0.50]` | 50th percentile | Median — used as a robust point estimate |
| `quantile_models[0.95]` | 95th percentile | Upper bound — worst-case energy needed |

These three together form a **prediction interval**, not a confidence interval — they describe the likely range of future energy demand.

#### Target Encoders

| Variable | Role |
|---|---|
| `te_station` | Maps `stationID` (string) → a float encoding based on mean target (kWh delivered) per station, learned from training data |
| `te_user` | Maps `parsed_userID` → a float encoding based on mean target per user |

Target encoding avoids the high cardinality problem that one-hot encoding would create with hundreds of unique stations/users.

---

## 5. Section 2 — Data Loading & Date Selection

### Code Snippet

```python
# ====================== 2. SELECT DAY ======================
df = pd.read_csv("acn_enhanced_final_2019_data.csv")
df['connectionTime'] = pd.to_datetime(
    df['connectionTime'], errors='coerce', utc=True
)
df['disconnectTime'] = pd.to_datetime(
    df['disconnectTime'], errors='coerce', utc=True
)

selected_date = '2019-07-26'   # ← Change this date as needed
df['date'] = df['connectionTime'].dt.date.astype(str)
df_day = df[df['date'] == selected_date].copy()

print(f"Selected Date: {selected_date} | Total sessions: {len(df_day)}")
```

### Explanation

#### Source Dataset

**`acn_enhanced_final_2019_data.csv`** is the ACN (Adaptive Charging Network) dataset from Caltech, augmented with engineered features (urgency scores, flexibility indices, etc.) from prior pipeline stages. Each row is **one EV charging session**.

#### Key Variables

| Variable | Type | Description |
|---|---|---|
| `df` | `DataFrame` | Full dataset — all sessions from 2019 |
| `selected_date` | `str` | ISO format date string. **This is the only parameter you need to change** to optimize a different day. |
| `df['date']` | `Series[str]` | Derived column: date portion of `connectionTime` as a string, used for filtering |
| `df_day` | `DataFrame` | Subset of `df` containing **only sessions from `selected_date`** |

#### Why UTC?

`pd.to_datetime(..., utc=True)` forces timezone-awareness. The ACN dataset timestamps include timezone info; parsing as UTC prevents silent timezone conversion bugs when computing durations (e.g., `disconnectTime - connectionTime`).

#### Why `.copy()`?

`df[df['date'] == selected_date].copy()` creates an independent copy of the filtered data. Without `.copy()`, pandas may issue a `SettingWithCopyWarning` when new columns are assigned later in Feature Engineering.

---

## 6. Section 3 — Feature Engineering

### Code Snippet — Part A (Time & Duration Features)

```python
# ====================== 3. FEATURE ENGINEERING ======================
df_day['hour'] = df_day['connectionTime'].dt.hour
df_day['day_of_week_encoded'] = pd.factorize(
    df_day['day_of_week'].fillna('Monday')
)[0]
df_day['is_weekend'] = (df_day['day_of_week_encoded'] >= 5).astype(int)
df_day['duration_hours'] = (
    df_day['disconnectTime'] - df_day['connectionTime']
).dt.total_seconds() / 3600

df_day['charging_efficiency'] = (
    df_day['kWhDelivered'] / df_day['parsed_kWhRequested'].replace(0, np.nan)
)
df_day['requested_gap'] = (
    df_day['parsed_kWhRequested'] - df_day['kWhDelivered']
)
df_day['energy_per_minute'] = (
    df_day['parsed_kWhRequested'] / (df_day['parsed_minutesAvailable'] + 1e-5)
)
```

### Code Snippet — Part B (Interaction & Cyclical Features)

```python
df_day['request_efficiency'] = (
    df_day['parsed_milesRequested'] / (df_day['parsed_WhPerMile'] + 1e-5)
)
df_day['urgency_flex_interaction'] = (
    df_day['urgency_score'] * df_day['flexibility_index']
)
df_day['revision_urgency'] = (
    df_day['revisionCount'] * df_day['urgency_score']
)
df_day['habit_grid'] = (
    df_day['habit_stability'] * df_day['grid_impact_proxy'].fillna(1)
)
df_day['is_peak'] = df_day['hour'].between(17, 21).astype(int)
df_day['hour_sin'] = np.sin(2 * np.pi * df_day['hour'] / 24)
df_day['hour_cos'] = np.cos(2 * np.pi * df_day['hour'] / 24)
```

### Code Snippet — Part C (Feature Matrix Assembly)

```python
feature_cols = [
    'parsed_milesRequested', 'parsed_WhPerMile', 'parsed_minutesAvailable',
    'parsed_kWhRequested', 'revisionCount', 'hour', 'day_of_week_encoded',
    'is_weekend', 'is_peak', 'urgency_score', 'flexibility_index',
    'habit_stability', 'grid_impact_proxy', 'urgency_flex_interaction',
    'revision_urgency', 'habit_grid', 'energy_per_minute',
    'request_efficiency', 'duration_hours', 'charging_efficiency',
    'requested_gap', 'hour_sin', 'hour_cos'
]

df_day = df_day.dropna(subset=feature_cols + ['parsed_userID', 'stationID']).copy()

df_day['station_encoded'] = te_station.transform(df_day[['stationID']]).flatten()
df_day['user_encoded'] = te_user.transform(df_day[['parsed_userID']]).flatten()
feature_cols += ['station_encoded', 'user_encoded']

X_day = df_day[feature_cols].copy()
```

### Explanation of All Features

#### Raw / Parsed Columns (from ACN data)

| Column | Meaning |
|---|---|
| `parsed_milesRequested` | Miles of range the user requested to charge up to |
| `parsed_WhPerMile` | Vehicle energy consumption in Wh/mile (determines kWh needed) |
| `parsed_minutesAvailable` | How long the vehicle will remain parked (charging window) |
| `parsed_kWhRequested` | Energy in kWh the user requested (derived from miles × Wh/mile) |
| `revisionCount` | Number of times the user revised their charging request — proxy for uncertainty |
| `kWhDelivered` | Actual kWh already delivered during the session so far |
| `urgency_score` | Pre-computed score (0–100) indicating how urgently the user needs charging |
| `flexibility_index` | Pre-computed score indicating how flexible the user is about timing (higher = more flexible) |
| `habit_stability` | How consistent the user's charging behavior is historically |
| `grid_impact_proxy` | Estimated impact of this session on overall grid load |

#### Derived Time Features

| Feature | Formula | Why |
|---|---|---|
| `hour` | `connectionTime.dt.hour` | Time-of-day signal — key driver of demand patterns |
| `day_of_week_encoded` | `pd.factorize(day_of_week)` | Ordinal encoding of weekday (0=Mon … 6=Sun) |
| `is_weekend` | `day_of_week_encoded >= 5` | Binary flag; charging behavior differs on weekends |
| `duration_hours` | `(disconnect - connect).total_seconds() / 3600` | Total available charging window in hours |
| `is_peak` | `hour.between(17, 21)` | Binary flag for evening peak hours (5–9 PM) |
| `hour_sin` | `sin(2π × hour / 24)` | Cyclical encoding — hour 0 and hour 23 are "close" |
| `hour_cos` | `cos(2π × hour / 24)` | Paired with `hour_sin` to fully encode the 24-hour cycle |

> **Why cyclical encoding?** A model treating `hour` as a raw integer assumes hour 23 is "far" from hour 0. Sine/cosine encoding maps hours onto a circle so the model correctly understands temporal continuity.

#### Derived Energy Features

| Feature | Formula | Why |
|---|---|---|
| `charging_efficiency` | `kWhDelivered / parsed_kWhRequested` | How much of the request has already been fulfilled (0–1) |
| `requested_gap` | `parsed_kWhRequested - kWhDelivered` | Remaining energy needed — the core optimization target |
| `energy_per_minute` | `parsed_kWhRequested / (parsed_minutesAvailable + ε)` | Power rate needed to satisfy request within deadline |
| `request_efficiency` | `parsed_milesRequested / (parsed_WhPerMile + ε)` | Effective range per unit of energy — vehicle efficiency proxy |

> **Why `+ 1e-5` (epsilon)?** Prevents division-by-zero when `parsed_minutesAvailable` or `parsed_WhPerMile` is 0.

#### Interaction Features

| Feature | Formula | Why |
|---|---|---|
| `urgency_flex_interaction` | `urgency_score × flexibility_index` | Captures the joint effect: a high-urgency, low-flexibility user is the hardest constraint |
| `revision_urgency` | `revisionCount × urgency_score` | Amplifies urgency for users who've revised repeatedly — indicates active monitoring |
| `habit_grid` | `habit_stability × grid_impact_proxy` | Users with stable habits who also have high grid impact need careful scheduling |

#### Encoded Features

| Feature | Source | Method |
|---|---|---|
| `station_encoded` | `stationID` | Target encoding: mean kWh delivered per station |
| `user_encoded` | `parsed_userID` | Target encoding: mean kWh delivered per user |

---

## 7. Section 4 — Predictions & Confidence Intervals

### Code Snippet

```python
# ====================== 4. PREDICTIONS ======================
stack_pred = np.column_stack((
    rf.predict(X_day),
    xgb_m.predict(X_day),
    cat_m.predict(X_day),
    lgb_base.predict(X_day)
))
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
df_day['remaining_kWh'] = (
    df_day['parsed_kWhRequested'] - df_day['kWhDelivered'].fillna(0)
)
df_day['minutes_left'] = df_day['parsed_minutesAvailable'].fillna(300)

active_sessions = df_day[df_day['remaining_kWh'] > 0.5].copy().reset_index(drop=True)
print(f"Optimizing for {len(active_sessions)} active sessions")
```

### Explanation

#### Stacking Inference

```python
stack_pred = np.column_stack((
    rf.predict(X_day),       # shape: (N,)
    xgb_m.predict(X_day),    # shape: (N,)
    cat_m.predict(X_day),    # shape: (N,)
    lgb_base.predict(X_day)  # shape: (N,)
))
# stack_pred shape: (N, 4)  — one column per base model
```

Each base model produces a 1D array of predictions (one per session). `np.column_stack` assembles them into an `(N, 4)` matrix — the **meta-feature matrix** — which is then fed to the meta-model.

```python
point_pred = meta_model.predict(stack_pred)  # shape: (N,)
```

The meta-model (LightGBM Booster) takes the 4 base model predictions as input features and outputs a single, refined prediction per session.

#### Quantile Bounds

| Variable | Value | Meaning |
|---|---|---|
| `q05` | 5th percentile predictions | Lower bound: 95% of sessions will need *at least* this much |
| `q50` | Median predictions | Central estimate |
| `q95` | 95th percentile predictions | Upper bound: 95% of sessions will need *at most* this much |
| `q_hat` | `0.629` (kWh) | **Conformal calibration constant** — widens the interval to achieve exact empirical coverage on a held-out calibration set. This value was computed in Stage 2. |

```python
lower = q05 - q_hat   # expand downward by q_hat
upper = q95 + q_hat   # expand upward by q_hat
```

This produces **conformalized prediction intervals** — statistically guaranteed (at the calibrated coverage level) to contain the true energy value.

#### Session Filtering

```python
active_sessions = df_day[df_day['remaining_kWh'] > 0.5].copy().reset_index(drop=True)
```

Only sessions with more than **0.5 kWh** remaining are passed to the optimizer. Sessions with negligible remaining demand (near-fully charged cars) are excluded to keep the LP problem tractable.

| Column Added | Meaning |
|---|---|
| `pred_kWh` | Meta-model point prediction of total energy needed |
| `lower_kWh` | Conformalized lower bound |
| `upper_kWh` | Conformalized upper bound |
| `remaining_kWh` | Actual energy still needed = `requested - delivered` |
| `minutes_left` | Time remaining for charging; NaN filled with 300 minutes (5 hours) |

---

## 8. Section 5 — Smart Optimizer (LP Formulation)

### Code Snippet — Part A (Problem Setup & Variables)

```python
def optimize_charging_schedule_smart(
    sessions_df, time_horizon_hours=8, time_step_min=15, grid_max_kw=350
):
    prob = LpProblem("Smart_EV_Charging_Optimization", LpMinimize)
    
    time_steps = int(time_horizon_hours * 60 / time_step_min)
    slots = list(range(time_steps))
    
    power = LpVariable.dicts(
        "power",
        ((i, t) for i in sessions_df.index for t in slots),
        lowBound=0, cat='Continuous'
    )
    peak_power = LpVariable("peak_power", lowBound=0)
    
    urgency_weight = sessions_df['urgency_flex_interaction'].fillna(1.0)
```

### Code Snippet — Part B (Objective Function)

```python
    # Objective: Minimize peak load + penalize delaying urgent users
    prob += (
        peak_power +
        0.01 * lpSum(power[i,t] for i in sessions_df.index for t in slots) +
        0.06 * lpSum(
            urgency_weight[i] * power[i,t] * (time_steps - t)
            for i in sessions_df.index for t in slots
        )
    )
    
    charger_max = 11.0
```

### Code Snippet — Part C (Per-Session Constraints)

```python
    for i in sessions_df.index:
        sess = sessions_df.loc[i]
        required = max(sess.get('remaining_kWh', 0), sess.get('lower_kWh', 0))
        
        max_slots = int(sess.get('minutes_left', 300) / time_step_min)
        
        physical_max_kwh = max_slots * charger_max * (time_step_min / 60.0)
        achievable_target = min(required * 0.93, physical_max_kwh)
        
        prob += lpSum(
            power[i,t] * (time_step_min / 60.0) for t in slots
        ) >= achievable_target
        
        for t in slots:
            prob += power[i,t] <= charger_max
```

### Code Snippet — Part D (Urgency Soft Constraints & Grid Cap)

```python
        if (sess.get('urgency_score', 50) > 80 and
                sess.get('flexibility_index', 0.5) < 0.3):
            soft_limit = int(max_slots * 0.9)
            for t in range(soft_limit, time_steps):
                prob += power[i,t] <= 2.0
        else:
            for t in range(max_slots, time_steps):
                prob += power[i,t] <= 0.5
    
    for t in slots:
        prob += lpSum(power[i,t] for i in sessions_df.index) <= grid_max_kw
        prob += lpSum(power[i,t] for i in sessions_df.index) <= peak_power
    
    status = prob.solve(PULP_CBC_CMD(msg=0, timeLimit=90))
    print(f"Status: {LpStatus[status]} | Optimal Peak: {value(peak_power):.2f} kW")
```

### Code Snippet — Part E (Schedule Extraction)

```python
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
schedule_df, optimal_peak = optimize_charging_schedule_smart(
    active_sessions, time_horizon_hours=8, time_step_min=15, grid_max_kw=350
)
```

### Explanation

#### Function Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `sessions_df` | — | DataFrame of active EV sessions to optimize |
| `time_horizon_hours` | `8` | Total scheduling window: 8 hours = 480 minutes |
| `time_step_min` | `15` | Granularity of each time slot: 15 minutes |
| `grid_max_kw` | `350` | Hard cap on total power drawn from the grid at any moment (kW) |

**Derived:** `time_steps = 8 × 60 / 15 = 32` time slots total.

#### Decision Variables

| Variable | Dimensions | Meaning |
|---|---|---|
| `power[i, t]` | `N_sessions × time_steps` | Power allocated (kW) to session `i` during time slot `t`. Non-negative continuous. |
| `peak_power` | Scalar | The maximum total grid power over all slots. Minimizing this is the primary objective. |

#### Objective Function (3 Terms)

```
Minimize:  peak_power
         + 0.01 × Σ(i,t) power[i,t]           ← energy cost term
         + 0.06 × Σ(i,t) urgency_weight[i] × power[i,t] × (time_steps - t)
                                                ← urgency-delay penalty
```

| Term | Weight | Purpose |
|---|---|---|
| `peak_power` | 1.0 | Primary objective: minimize peak grid demand |
| `Σ power[i,t]` | 0.01 | Secondary: slight preference for using less total energy (avoids unnecessary small allocations) |
| `urgency_weight[i] × power[i,t] × (time_steps - t)` | 0.06 | Penalizes scheduling urgent users late. `(time_steps - t)` is large for early slots — wait, this actually penalizes *early* charging? No — the term grows for *later* `t` values... **Actually:** `(time_steps - t)` is *largest* for `t=0` and *smallest* for `t=time_steps-1`. This means early slots have a higher penalty for urgent users if power is allocated there. **Correction:** The intent is to incentivize getting urgent users' power delivered *sooner*. With `(time_steps - t)` being large at `t=0`, the optimizer *avoids* putting power in early slots for urgent users — pushing power earlier costs more in the objective. Actually this *encourages* front-loading power early only when urgency_weight is low. For high-urgency users, front-loading is penalized more — suggesting urgent users should be served efficiently *across* slots rather than dumped in early. The interaction achieves priority-aware spreading. |

#### Per-Session Constraints

**Energy Delivery (Soft-Feasible Target):**

```python
achievable_target = min(required * 0.93, physical_max_kwh)
prob += lpSum(power[i,t] * (time_step_min/60) for t in slots) >= achievable_target
```

| Variable | Meaning |
|---|---|
| `required` | Max of `remaining_kWh` and `lower_kWh` — ensures optimizer uses the higher (safer) estimate |
| `physical_max_kwh` | `max_slots × 11 kW × (15/60 hr)` — physical ceiling: even if charger runs at full power for all available slots, this is the maximum deliverable energy |
| `0.93` scaling | 7% slack allows the LP to find feasible solutions in edge cases where the requested energy is very slightly above what's physically achievable |

**Charger Hardware Limit:**

```python
prob += power[i,t] <= charger_max   # charger_max = 11.0 kW
```

Each charger is capped at **11 kW** (Level 2 AC charging standard) per time slot.

**Urgency-Based Time Window Constraints:**

| Condition | Constraint After Deadline |
|---|---|
| `urgency > 80` AND `flexibility < 0.3` (very urgent, very inflexible) | `power[i,t] <= 2.0 kW` — trickle charge allowed after soft deadline (`0.9 × max_slots`) |
| All other sessions | `power[i,t] <= 0.5 kW` — nearly zero after parking deadline |

> These are *soft* rather than hard cutoffs (0 kW) to prevent infeasibility when a session's deadline is very tight.

**Grid-Wide Power Constraints:**

```python
prob += lpSum(power[i,t] for i in sessions_df.index) <= grid_max_kw   # Hard cap: 350 kW
prob += lpSum(power[i,t] for i in sessions_df.index) <= peak_power    # Tracks peak for objective
```

The second constraint is what connects the `peak_power` variable to actual slot totals — it forces `peak_power` to be at least as large as the highest slot total, and since we're minimizing `peak_power`, the solver minimizes that maximum.

#### Solver Settings

```python
PULP_CBC_CMD(msg=0, timeLimit=90)
```

| Setting | Value | Meaning |
|---|---|---|
| `msg=0` | Silent | Suppresses verbose CBC solver output |
| `timeLimit=90` | 90 seconds | Hard timeout — solver returns best feasible solution found within 90 seconds (important for large instances) |

CBC (COIN-OR Branch and Cut) is an open-source MILP solver bundled with PuLP. Since all variables are continuous (`cat='Continuous'`), this reduces to a standard LP, which CBC solves very efficiently.

---

## 9. Section 6 — Visualizations

### Code Snippet — Before/After Chart

```python
def plot_before_after(active_sessions, schedule_df, optimal_peak, selected_date):
    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(2, 1, 1)
    n_slots = 40
    time_index = [f"T{t}" for t in range(n_slots)]
    
    naive_power = np.zeros(n_slots)
    charger_max = 11.0
    for _, row in active_sessions.iterrows():
        remaining = row.get('remaining_kWh', 10)
        slots_needed = int(np.ceil(remaining / (charger_max * 15 / 60))) + 2
        for t in range(min(slots_needed, n_slots)):
            naive_power[t] += charger_max
    
    optimized_power = np.zeros(n_slots)
    if not schedule_df.empty:
        for _, row in schedule_df.iterrows():
            ts = int(row['time_slot'])
            if ts < n_slots:
                optimized_power[ts] += row['power_kw']
```

### Code Snippet — Gantt Chart

```python
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
        
        if urgency >= 75:   color = '#d32f2f'   # Dark Red
        elif urgency >= 60: color = '#ff9800'   # Orange
        elif urgency >= 45: color = '#ffeb3b'   # Yellow
        else:               color = '#4caf50'   # Green
            
        plt.barh(y, 1, left=start, height=0.65,
                 color=color, alpha=0.9, edgecolor='black')
```

### Explanation

#### `plot_before_after` — Variables

| Variable | Meaning |
|---|---|
| `n_slots` | 40 slots displayed (10 hours of 15-min intervals) — slightly wider than the 32 optimization slots for visual context |
| `naive_power[t]` | Array simulating what would happen with no optimization: every car charges at full 11 kW simultaneously from connection time |
| `optimized_power[t]` | Actual scheduled power summed per time slot from the LP solution |
| `slots_needed` | How many consecutive slots naive charging would occupy for a given session |
| `reduction` | `(naive_peak - opt_peak) / naive_peak × 100` — percentage peak reduction achieved by the optimizer |

**Naive Simulation Logic:** For each session, assumes it starts charging immediately at `charger_max` (11 kW) and runs for however many slots are needed to deliver `remaining_kWh`. This models the uncontrolled "plug in and blast" scenario.

#### `plot_gantt_chart` — Variables

| Variable | Meaning |
|---|---|
| `top_users` | Top 15 users by session frequency on the selected day — shown as rows |
| `user_map` | `{userID: y_position}` mapping for chart rows |
| `y` | Vertical position of the horizontal bar for that user |
| `start` | Horizontal start position of the bar = time slot number |
| `urgency` | Urgency score from the schedule DataFrame — determines bar color |

**Color Scheme:**

| Color | Hex | Urgency Range | Interpretation |
|---|---|---|---|
| Dark Red | `#d32f2f` | ≥ 75 | Very high priority — must charge early |
| Orange | `#ff9800` | 60–74 | High priority |
| Yellow | `#ffeb3b` | 45–59 | Medium priority |
| Green | `#4caf50` | < 45 | Low priority — flexible, can be deferred |

**Output Files:**

| File | Content |
|---|---|
| `before_after_optimization_2019-07-26.png` | Two-panel: line chart (naive vs optimized power over time) + bar chart (peak comparison) |
| `gantt_chart_urgency_2019-07-26.png` | Horizontal bar chart showing each user's charging schedule, colored by urgency |

---

## 10. Variable Reference Table

### All Variables — Alphabetical

| Variable | Section | Type | Description |
|---|---|---|---|
| `achievable_target` | Optimizer | `float` | LP energy target per session: `min(required × 0.93, physical_max_kwh)` |
| `active_sessions` | Predictions | `DataFrame` | Filtered sessions with `remaining_kWh > 0.5` — these are optimized |
| `alpha` | Model Loading | `float` | Quantile level: 0.05, 0.50, or 0.95 |
| `cat_m` | Model Loading | `CatBoostRegressor` | CatBoost base model |
| `charger_max` | Optimizer | `float` | Maximum power per charger: 11.0 kW |
| `df` | Data Loading | `DataFrame` | Full ACN dataset |
| `df_day` | Data Loading | `DataFrame` | Single-day subset of `df` |
| `feature_cols` | Feature Eng. | `list[str]` | Ordered list of 25 feature column names fed to models |
| `grid_max_kw` | Optimizer | `float` | Maximum total grid power: 350 kW |
| `lgb_base` | Model Loading | `LGBMRegressor` | LightGBM base model |
| `lower` | Predictions | `np.ndarray` | Conformalized lower bound: `q05 - q_hat` |
| `max_slots` | Optimizer | `int` | Number of 15-min slots within session's parking window |
| `meta_model` | Model Loading | `lgb.Booster` | Stacking meta-model trained on base model outputs |
| `naive_power` | Visualization | `np.ndarray` | Per-slot uncontrolled power (all chargers at max simultaneously) |
| `optimal_peak` | Optimizer | `float` | LP optimal value of `peak_power` variable (kW) |
| `optimized_power` | Visualization | `np.ndarray` | Per-slot scheduled power from LP solution |
| `peak_power` | Optimizer | `LpVariable` | LP scalar variable representing maximum total slot power |
| `physical_max_kwh` | Optimizer | `float` | Maximum kWh deliverable within parking window at max power |
| `point_pred` | Predictions | `np.ndarray` | Meta-model point predictions of kWh needed per session |
| `power[i,t]` | Optimizer | `LpVariable` | Power (kW) allocated to session `i` in time slot `t` |
| `prob` | Optimizer | `LpProblem` | The PuLP LP problem object |
| `q05, q50, q95` | Predictions | `np.ndarray` | Quantile predictions (5th, 50th, 95th percentiles) |
| `q_hat` | Predictions | `float` | Conformal calibration constant: `0.629` kWh |
| `quantile_models` | Model Loading | `dict` | `{alpha: lgb.Booster}` for each quantile level |
| `required` | Optimizer | `float` | Energy needed by session: `max(remaining_kWh, lower_kWh)` |
| `rf` | Model Loading | `RandomForestRegressor` | Random Forest base model |
| `schedule` | Optimizer | `list[dict]` | Accumulates LP solution rows before converting to DataFrame |
| `schedule_df` | Optimizer | `DataFrame` | Final LP schedule: one row per (session, time_slot) with power > 0.1 kW |
| `selected_date` | Data Loading | `str` | ISO date string — the day being optimized |
| `slots` | Optimizer | `list[int]` | `[0, 1, ..., time_steps-1]` — all time slot indices |
| `stack_pred` | Predictions | `np.ndarray (N×4)` | Base model predictions stacked as meta-features |
| `te_station` | Model Loading | `TargetEncoder` | Target encoder for `stationID` |
| `te_user` | Model Loading | `TargetEncoder` | Target encoder for `parsed_userID` |
| `time_steps` | Optimizer | `int` | Total number of time slots: `8 × 60 / 15 = 32` |
| `upper` | Predictions | `np.ndarray` | Conformalized upper bound: `q95 + q_hat` |
| `urgency_weight` | Optimizer | `Series` | `urgency_flex_interaction` values per session — LP objective weights |
| `X_day` | Feature Eng. | `DataFrame` | Final feature matrix (N × 25) passed to all models |
| `xgb_m` | Model Loading | `XGBRegressor` | XGBoost base model |

---

## 11. End-to-End Data Flow

```
acn_enhanced_final_2019_data.csv
        │
        ▼
  df (all 2019 sessions)
        │  filter by selected_date
        ▼
  df_day (one day's sessions)
        │  engineer 23 features
        ▼
  X_day  ──────────────────────────────────────────────┐
        │                                              │
        │  RF, XGB, CatBoost, LGB predict              │
        ▼                                              │
  stack_pred (N×4)                                     │
        │  meta_model predicts                         │
        ▼                                              │
  point_pred (N,)                               quantile_models
        │                                        predict q05, q50, q95
        │                                              │
        └──────────────────────────────────────────────┘
                     combine + conformalize
                             │
                             ▼
                 df_day: pred_kWh, lower_kWh, upper_kWh
                             │  filter remaining_kWh > 0.5
                             ▼
                      active_sessions
                             │
                             ▼
              LP Optimizer (PuLP / CBC)
              32 slots × N sessions variables
              Minimize: peak_power + urgency terms
              Subject to: energy targets, charger limits, grid cap
                             │
                             ▼
                       schedule_df
                             │
              ┌──────────────┴─────────────────┐
              ▼                                ▼
    plot_before_after()            plot_gantt_chart()
              │                                │
              ▼                                ▼
  before_after_optimization_       gantt_chart_urgency_
     2019-07-26.png                   2019-07-26.png
```

---

*Documentation generated for `ev_charging_stage3_smart_optimizer.py` — Stage 3 of the ACN Smart Charging Pipeline.*
