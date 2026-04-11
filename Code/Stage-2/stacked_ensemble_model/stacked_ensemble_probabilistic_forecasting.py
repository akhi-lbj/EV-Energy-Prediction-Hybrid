import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from datetime import datetime
import warnings
import joblib
warnings.filterwarnings('ignore')

print("🚀 Loading dataset for Ablation Study (exact same split as v15)...")
if os.path.exists("meta_model_metrics_per_iteration.csv"):
    os.remove("meta_model_metrics_per_iteration.csv")
df = pd.read_csv("acn_enhanced_final_2019_data.csv")

# --- Temporal features (same as v15) ---
df['connectionTime'] = pd.to_datetime(df['connectionTime'], errors='coerce', utc=True)
df['hour_of_day'] = df['connectionTime'].dt.hour
df['month'] = df['connectionTime'].dt.month
df['day_of_week_encoded'] = pd.factorize(df['day_of_week'].fillna('Monday'))[0]

# --- ALL derived features (we'll selectively drop behavioral ones) ---
df['energy_per_minute'] = df['parsed_kWhRequested'] / (df['parsed_minutesAvailable'] + 1e-5)
df['request_efficiency'] = df['parsed_milesRequested'] / (df['parsed_WhPerMile'] + 1e-5)
df['urgency_flex_interaction'] = df['urgency_score'] * df['flexibility_index']
df['revision_urgency'] = df['revisionCount'] * df['urgency_score']
df['habit_grid'] = df['habit_stability'] * df['grid_impact_proxy']

# --- Exact same 50:10:40 month-stratified split ---
np.random.seed(42)
test_indices, val_indices = [], []
for month in range(1, 13):
    month_df = df[df['month'] == month]
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
df_val   = df[val_mask].copy()
df_test  = df[test_mask].copy()

print(f"Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

# --- Target Encoding (train-only) ---
te_station = TargetEncoder(target_type='continuous', random_state=42)
te_user    = TargetEncoder(target_type='continuous', random_state=42)

df_train['station_encoded'] = te_station.fit_transform(df_train[['stationID']], df_train['kWhDelivered']).flatten()
df_train['user_encoded']    = te_user.fit_transform(df_train[['parsed_userID']], df_train['kWhDelivered']).flatten()

df_val['station_encoded']  = te_station.transform(df_val[['stationID']]).flatten()
df_val['user_encoded']     = te_user.transform(df_val[['parsed_userID']]).flatten()
df_test['station_encoded'] = te_station.transform(df_test[['stationID']]).flatten()
df_test['user_encoded']    = te_user.transform(df_test[['parsed_userID']]).flatten()

# Base features (always kept)
base_features = [
    'parsed_milesRequested', 'parsed_WhPerMile', 'parsed_minutesAvailable',
    'parsed_kWhRequested', 'revisionCount', 'hour_of_day', 'day_of_week_encoded',
    'is_weekend', 'urgency_score', 'flexibility_index', 'habit_stability',
    'grid_impact_proxy', 'request_efficiency', 'station_encoded', 'user_encoded'
]

behavioral_features = ['urgency_flex_interaction', 'revision_urgency', 'habit_grid', 'energy_per_minute']

def run_ablation(with_behavioral=True):
    suffix = "WITH" if with_behavioral else "WITHOUT"
    print(f"\n=== Ablation: {suffix} 4 Behavioral Features ===")
    
    feat_cols = base_features + (behavioral_features if with_behavioral else [])
    
    X_train = df_train[feat_cols]
    y_train = df_train['kWhDelivered']
    X_val   = df_val[feat_cols]
    y_val   = df_val['kWhDelivered']
    X_test  = df_test[feat_cols]
    y_test  = df_test['kWhDelivered']
    
    # Same stacked ensemble as v15 (RF + XGB + Cat + HistGB → LGB meta)
    rf = RandomForestRegressor(n_estimators=300, max_depth=9, min_samples_leaf=5, max_features=0.75, random_state=42, n_jobs=-1)
    xgb_model = xgb.XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.025, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=1.5, random_state=42, tree_method='hist')
    cat_model = cb.CatBoostRegressor(iterations=600, depth=8, learning_rate=0.03, l2_leaf_reg=4, random_seed=42, verbose=0)
    hist_model = HistGradientBoostingRegressor(max_iter=600, max_depth=8, learning_rate=0.03, random_state=42)
    
    rf.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    cat_model.fit(X_train, y_train)
    hist_model.fit(X_train, y_train)
    
    stack_train = np.column_stack((rf.predict(X_train), xgb_model.predict(X_train), cat_model.predict(X_train), hist_model.predict(X_train)))
    stack_val   = np.column_stack((rf.predict(X_val),   xgb_model.predict(X_val),   cat_model.predict(X_val),   hist_model.predict(X_val)))
    stack_test  = np.column_stack((rf.predict(X_test),  xgb_model.predict(X_test),  cat_model.predict(X_test),  hist_model.predict(X_test)))
    
    lgb_train_ds = lgb.Dataset(stack_train, label=y_train)
    lgb_val_ds   = lgb.Dataset(stack_val, label=y_val)
    lgb_test_ds  = lgb.Dataset(stack_test, label=y_test)
    
    meta_params = {'objective': 'regression', 'metric': ['rmse', 'mae'], 'num_leaves': 63, 'learning_rate': 0.015, 'feature_fraction': 0.9, 'bagging_fraction': 0.9, 'lambda_l1': 3.0, 'lambda_l2': 3.0, 'verbose': -1, 'random_state': 42}
    
    def lgb_r2_score(preds, dmatrix):
        labels = dmatrix.get_label()
        return 'r2', r2_score(labels, preds), True

    evals_meta = {}
    meta_model = lgb.train(
        meta_params, lgb_train_ds, num_boost_round=1500, 
        valid_sets=[lgb_train_ds, lgb_val_ds, lgb_test_ds], 
        valid_names=['train', 'val', 'test'],
        feval=lgb_r2_score,
        callbacks=[lgb.record_evaluation(evals_meta), lgb.early_stopping(150, verbose=False)]
    )
    
    # Save per-iteration metrics for meta model
    iterations = len(evals_meta['train']['rmse'])
    history_df = pd.DataFrame({
        'experiment': f'Ablation_{suffix}_Behavioral [LightGBM Meta]',
        'iteration': range(1, iterations + 1),
        'train_rmse': evals_meta['train']['rmse'],
        'train_mae': evals_meta['train']['l1'],
        'train_r2': evals_meta['train']['r2'],
        'val_rmse': evals_meta['val']['rmse'],
        'val_mae': evals_meta['val']['l1'],
        'val_r2': evals_meta['val']['r2'],
        'test_rmse': evals_meta['test']['rmse'],
        'test_mae': evals_meta['test']['l1'],
        'test_r2': evals_meta['test']['r2'],
    })
    log_file = "meta_model_metrics_per_iteration.csv"
    history_df.to_csv(log_file, mode='a' if os.path.exists(log_file) else 'w', header=not os.path.exists(log_file), index=False)
    
    final_train = meta_model.predict(stack_train)
    final_val   = meta_model.predict(stack_val)
    final_test  = meta_model.predict(stack_test)
    
    train_mae = mean_absolute_error(y_train, final_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, final_train))
    train_r2 = r2_score(y_train, final_train)

    val_mae = mean_absolute_error(y_val, final_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, final_val))
    val_r2 = r2_score(y_val, final_val)

    test_mae = mean_absolute_error(y_test, final_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, final_test))
    test_r2 = r2_score(y_test, final_test)
    
    print(f"  Train MAE: {train_mae:.3f} kWh | RMSE: {train_rmse:.3f} | R²: {train_r2:.4f}")
    print(f"  Val   MAE: {val_mae:.3f} kWh | RMSE: {val_rmse:.3f} | R²: {val_r2:.4f}")
    print(f"  Test  MAE: {test_mae:.3f} kWh | RMSE: {test_rmse:.3f} | R²: {test_r2:.4f}")
    
    # Log
    pd.DataFrame([{
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'experiment': f'Ablation_{suffix}_Behavioral',
        'train_mae': train_mae, 'train_rmse': train_rmse, 'train_r2': train_r2,
        'val_mae': val_mae, 'val_rmse': val_rmse, 'val_r2': val_r2,
        'test_mae': test_mae, 'test_rmse': test_rmse, 'test_r2': test_r2,
        'note': 'Same 50:10:40 split + same stacking as v15'
    }]).to_csv("ablation_results_log.csv", mode='a', header=not os.path.exists("ablation_results_log.csv"), index=False)
    
    if with_behavioral:
        print("\n💾 Saving models from 'WITH behavioral' iteration...")
        os.makedirs("ensemble_models_v16", exist_ok=True)
        joblib.dump(rf, "ensemble_models_v16/rf_base.pkl")
        joblib.dump(xgb_model, "ensemble_models_v16/xgb_base.pkl")
        joblib.dump(cat_model, "ensemble_models_v16/cat_base.pkl")
        joblib.dump(hist_model, "ensemble_models_v16/hist_base.pkl")
        meta_model.save_model("ensemble_models_v16/meta_lgb.txt")
        joblib.dump(te_station, "ensemble_models_v16/te_station.pkl")
        joblib.dump(te_user, "ensemble_models_v16/te_user.pkl")
        print("✅ Models and TargetEncoders saved in 'ensemble_models_v16/'")
        
    return test_mae, test_rmse, test_r2

# Run both
res_with = run_ablation(with_behavioral=True)
res_without = run_ablation(with_behavioral=False)

print("\n=== ABLATION SUMMARY (Test Set) ===")
print(f"WITH  behavioral → Test MAE: {res_with[0]:.3f} | R²: {res_with[2]:.4f}")
print(f"WITHOUT behavioral → Test MAE: {res_without[0]:.3f} | R²: {res_without[2]:.4f}")
print(f"Delta Test MAE: {res_with[0] - res_without[0]:+.3f} kWh (negative = improvement with features)")

# ====================== PROBABILISTIC FORECASTING LAYER ======================
# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
print("\n🚀 Starting Probabilistic Forecasting Layer (QRF + QEGBR-style)...")
print("Using exact same features & split as your stacked ensemble (WITH behavioral features)")

# === CRITICAL FIX: Force WITH behavioral features (matches your paper) ===
feat_cols = base_features + behavioral_features   # ←←← THIS IS THE FIX

X_train = df_train[feat_cols].copy()
y_train = df_train['kWhDelivered'].copy()
X_val   = df_val[feat_cols].copy()
y_val   = df_val['kWhDelivered'].copy()
X_test  = df_test[feat_cols].copy()
y_test  = df_test['kWhDelivered'].copy()

# Create folder
os.makedirs("prob_models", exist_ok=True)

# ====================== 1. LightGBM Quantile Models (QRF-style) ======================
quantiles = [0.05, 0.50, 0.95]
lgb_quantile_models = {}

for alpha in quantiles:
    print(f"   Training LightGBM Quantile α = {alpha:.2f} ...")
    params = {
        'objective': 'quantile',
        'alpha': alpha,
        'metric': 'quantile',
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'lambda_l1': 3.0,
        'lambda_l2': 3.0,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    evals_lgb = {}
    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds   = lgb.Dataset(X_val, label=y_val)
    test_ds  = lgb.Dataset(X_test, label=y_test)
    model = lgb.train(params, train_ds, num_boost_round=1200,
                      valid_sets=[train_ds, val_ds, test_ds],
                      valid_names=['train', 'val', 'test'],
                      callbacks=[lgb.record_evaluation(evals_lgb), lgb.early_stopping(100, verbose=False)])
    
    lgb_quantile_models[alpha] = model
    model.save_model(f"prob_models/lgb_quantile_alpha_{int(alpha*100)}.txt")

    # Log iterations for LightGBM Quantile
    iter_df = pd.DataFrame({
        'experiment': f'Probabilistic_Total [LightGBM Quantile alpha_{alpha}]',
        'iteration': range(1, len(evals_lgb['train']['quantile']) + 1),
        'train_rmse': evals_lgb['train']['quantile'],
        'val_rmse': evals_lgb['val']['quantile'],
        'test_rmse': evals_lgb['test']['quantile'],
    })
    iter_df.to_csv("meta_model_metrics_per_iteration.csv", mode='a', header=False, index=False)

# ====================== 2. XGBoost Quantile Models (QEGBR-style) ======================
xgb_quantile_models = {}
for alpha in quantiles:
    print(f"   Training XGBoost Quantile α = {alpha:.2f} ...")
    params = {
        'objective': 'reg:quantileerror',
        'quantile_alpha': alpha,
        'learning_rate': 0.025,
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 3.0,
        'reg_alpha': 1.5,
        'random_state': 42,
        'tree_method': 'hist',
        'n_jobs': -1
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val, label=y_val)
    dtest  = xgb.DMatrix(X_test, label=y_test)
    evals_xgb_dict = {}
    model = xgb.train(params, dtrain, num_boost_round=800,
                      early_stopping_rounds=80,
                      evals=[(dtrain, 'train'), (dval, 'val'), (dtest, 'test')],
                      evals_result=evals_xgb_dict,
                      verbose_eval=False)
    
    xgb_quantile_models[alpha] = model
    model.save_model(f"prob_models/xgb_quantile_alpha_{int(alpha*100)}.model")

    # Log iterations for XGBoost Quantile
    iter_df = pd.DataFrame({
        'experiment': f'Probabilistic_Total [XGBoost Quantile alpha_{alpha}]',
        'iteration': range(1, len(evals_xgb_dict['train']['quantile']) + 1),
        'train_rmse': evals_xgb_dict['train']['quantile'],
        'val_rmse': evals_xgb_dict['val']['quantile'],
        'test_rmse': evals_xgb_dict['test']['quantile'],
    })
    iter_df.to_csv("meta_model_metrics_per_iteration.csv", mode='a', header=False, index=False)

# ====================== 3. Evaluation on All Sets ======================
print("\n📊 Evaluating Probabilistic Models on Train, Val, Test Sets...")

# Predictions on all datasets (using median: alpha=0.50)
lgb_train_pred = lgb_quantile_models[0.50].predict(X_train)
lgb_val_pred   = lgb_quantile_models[0.50].predict(X_val)
lgb_test_pred  = lgb_quantile_models[0.50].predict(X_test)

dtrain_eval = xgb.DMatrix(X_train)
dval_eval   = xgb.DMatrix(X_val)
dtest_eval  = xgb.DMatrix(X_test)
xgb_train_pred = xgb_quantile_models[0.50].predict(dtrain_eval)
xgb_val_pred   = xgb_quantile_models[0.50].predict(dval_eval)
xgb_test_pred  = xgb_quantile_models[0.50].predict(dtest_eval)

# Test PI metrics
lgb_lower  = lgb_quantile_models[0.05].predict(X_test)
lgb_upper  = lgb_quantile_models[0.95].predict(X_test)
xgb_lower  = xgb_quantile_models[0.05].predict(dtest_eval)
xgb_upper  = xgb_quantile_models[0.95].predict(dtest_eval)

def pi_metrics(y_true, lower, upper, nominal_coverage=0.90):
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    width = np.mean(upper - lower)
    return coverage, width

lgb_cov, lgb_width = pi_metrics(y_test, lgb_lower, lgb_upper)
xgb_cov, xgb_width = pi_metrics(y_test, xgb_lower, xgb_upper)

print("\n=== PROBABILISTIC RESULTS (Test Set) ===")
print(f"LightGBM Quantile (QRF-style) → 90% PI Coverage: {lgb_cov:.1%} | Avg Width: {lgb_width:.3f} kWh")
print(f"XGBoost Quantile (QEGBR-style) → 90% PI Coverage: {xgb_cov:.1%} | Avg Width: {xgb_width:.3f} kWh")

# Compute metrics
def calc_metrics(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        r2_score(y_true, y_pred)
    )

lgb_train_mae, lgb_train_rmse, lgb_train_r2 = calc_metrics(y_train, lgb_train_pred)
lgb_val_mae, lgb_val_rmse, lgb_val_r2 = calc_metrics(y_val, lgb_val_pred)
lgb_test_mae, lgb_test_rmse, lgb_test_r2 = calc_metrics(y_test, lgb_test_pred)

xgb_train_mae, xgb_train_rmse, xgb_train_r2 = calc_metrics(y_train, xgb_train_pred)
xgb_val_mae, xgb_val_rmse, xgb_val_r2 = calc_metrics(y_val, xgb_val_pred)
xgb_test_mae, xgb_test_rmse, xgb_test_r2 = calc_metrics(y_test, xgb_test_pred)


print("\nMedian Performance across Sets:")
print(f"LightGBM → Train MAE: {lgb_train_mae:.3f} | Val MAE: {lgb_val_mae:.3f} | Test MAE: {lgb_test_mae:.3f}")
print(f"XGBoost  → Train MAE: {xgb_train_mae:.3f} | Val MAE: {xgb_val_mae:.3f} | Test MAE: {xgb_test_mae:.3f}")

# Log results
pd.DataFrame([{
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'experiment': 'Probabilistic_Layer',
    'lgb_90_coverage': lgb_cov,
    'lgb_avg_width': lgb_width,
    'xgb_90_coverage': xgb_cov,
    'xgb_avg_width': xgb_width,
    'lgb_train_mae': lgb_train_mae, 'lgb_train_rmse': lgb_train_rmse, 'lgb_train_r2': lgb_train_r2,
    'lgb_val_mae': lgb_val_mae, 'lgb_val_rmse': lgb_val_rmse, 'lgb_val_r2': lgb_val_r2,
    'lgb_test_mae': lgb_test_mae, 'lgb_test_rmse': lgb_test_rmse, 'lgb_test_r2': lgb_test_r2,
    'xgb_train_mae': xgb_train_mae, 'xgb_train_rmse': xgb_train_rmse, 'xgb_train_r2': xgb_train_r2,
    'xgb_val_mae': xgb_val_mae, 'xgb_val_rmse': xgb_val_rmse, 'xgb_val_r2': xgb_val_r2,
    'xgb_test_mae': xgb_test_mae, 'xgb_test_rmse': xgb_test_rmse, 'xgb_test_r2': xgb_test_r2,
    'note': 'Same features & split as stacked ensemble (WITH behavioral)'
}]).to_csv("probabilistic_results_log.csv", mode='a', header=not os.path.exists("probabilistic_results_log.csv"), index=False)

print("\n✅ Probabilistic models saved in folder 'prob_models/'")
print("   → lgb_quantile_alpha_5.txt, 50.txt, 95.txt")
print("   → xgb_quantile_alpha_5.model, 50.model, 95.model")
print("   → Ready to combine with your stacked ensemble in the optimization layer!")