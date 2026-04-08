# Changes from Baseline to Latest Commit

This document lists all commits from **Making the Baseline** (interpreted as the first baseline commit) to the latest commit on this branch, in chronological order.

## Commit Range
- Start (Baseline): `a1ed2d2` — **Added_Baseline_model_From_Previous_Research_Work**
- End (Latest): `81070a1` — **Stacked_Ensemble_Updated**

## Step-by-Step Changes

1. **`a1ed2d2` — Added_Baseline_model_From_Previous_Research_Work**
   - Added: `Code/Stage-2/baseline_model/baseline_xgboost_ev_energy.py`
   - Summary: Introduced the baseline XGBoost EV energy prediction script.
   - Diff size: 1 file changed, 144 insertions.

2. **`6082d16` — destination_update_baseline**
   - Modified: `Code/Stage-2/baseline_model/baseline_xgboost_ev_energy.py`
   - Summary: Updated baseline script destination/path-related logic.
   - Diff size: 1 file changed, 1 insertion, 1 deletion.

3. **`65ad10b` — update**
   - Renamed/Moved:
     - `Code/Stage-2/dataset/acn_enhanced_final_2019_data.csv` → `Code/Stage-2/baseline_model/acn_enhanced_final_2019_data.csv`
   - Summary: Relocated the dataset into the baseline model folder.
   - Diff size: 1 file changed (rename only).

4. **`b8517d1` — update**
   - Modified: `Code/Stage-2/baseline_model/baseline_xgboost_ev_energy.py`
   - Summary: Small follow-up adjustment to baseline script.
   - Diff size: 1 file changed, 1 insertion, 1 deletion.

5. **`2cdb669` — Implemented_XGBoost_Baseline_with_results**
   - Added:
     - `Code/Stage-2/baseline_model/baseline_results_log.csv`
     - `Code/Stage-2/baseline_model/xgboost_models/day_le.pkl`
     - `Code/Stage-2/baseline_model/xgboost_models/grid_le.pkl`
     - `Code/Stage-2/baseline_model/xgboost_models/scaler_baseline_stratified.pkl`
     - `Code/Stage-2/baseline_model/xgboost_models/xgboost_baseline_stratified.json`
   - Modified:
     - `Code/Stage-2/baseline_model/baseline_xgboost_ev_energy.py`
   - Summary: Finalized baseline training/results pipeline and added saved model artifacts.
   - Diff size: 6 files changed, 119 insertions, 81 deletions.

6. **`cf39650` — Start_of_stacked_ensemble_model**
   - Added:
     - `Code/Stage-2/stacked_ensemble_model/acn_enhanced_final_2019_data.csv`
     - `Code/Stage-2/stacked_ensemble_model/baseline_results_log.csv`
     - `Code/Stage-2/stacked_ensemble_model/stacked_ensemble_ev_energy.py`
   - Summary: Started stacked ensemble phase with initial script and required data/log files.
   - Diff size: 3 files changed, 8902 insertions.

7. **`fe6d729` — Stacked_Ensemble**
   - Added:
     - `Code/Stage-2/stacked_ensemble_model/ablation_results_log.csv`
     - `Code/Stage-2/stacked_ensemble_model/catboost_info/...` (training metadata files)
     - `Code/Stage-2/stacked_ensemble_model/ensemble_models_v16/...` (saved ensemble model artifacts)
     - `Code/Stage-2/stacked_ensemble_model/improved_stacked_ensemble_v7.py`
     - `Code/Stage-2/stacked_ensemble_model/improved_stacked_ensemble_v8.py`
     - `Code/Stage-2/stacked_ensemble_model/improved_stacked_ensemble_v15.py`
     - `Code/Stage-2/stacked_ensemble_model/improved_stacked_ensemble_v16.py`
   - Modified:
     - `Code/Stage-2/stacked_ensemble_model/baseline_results_log.csv`
   - Summary: Major stacked ensemble implementation with iterative script versions, artifacts, and ablation outputs.
   - Diff size: 17 files changed, 31,159 insertions.

8. **`81070a1` — Stacked_Ensemble_Updated**
   - Deleted:
     - `Code/Stage-2/stacked_ensemble_model/improved_stacked_ensemble_v15.py`
     - `Code/Stage-2/stacked_ensemble_model/improved_stacked_ensemble_v7.py`
     - `Code/Stage-2/stacked_ensemble_model/improved_stacked_ensemble_v8.py`
     - `Code/Stage-2/stacked_ensemble_model/stacked_ensemble_ev_energy.py`
   - Summary: Cleaned up older/alternate stacked ensemble script versions, keeping the newer workflow.
   - Diff size: 4 files changed, 727 deletions.

---

## High-Level Evolution
- Started with a baseline XGBoost model implementation.
- Refined baseline setup and moved dataset into baseline workspace.
- Added baseline results and serialized baseline model components.
- Initiated stacked ensemble modeling with separate data/log setup.
- Expanded stacked ensemble into multiple improved versions with saved artifacts and ablation outputs.
- Removed older intermediate stacked scripts in the latest update.
