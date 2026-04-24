# EV Energy Demand Prediction: A Hybrid & Probabilistic Approach

A comprehensive machine learning project aimed at predicting the energy consumption of Electric Vehicles (EVs) using a Stacked Ensemble architecture. The model incorporates behavioral features and provides robust probabilistic forecasting with Prediction Intervals (PI).

## 📌 Overview

Accurately predicting EV charging demand is critical for grid stability and efficient infrastructure utilization. This project develops a predictive model that forecasts the exact energy (kWh) required for an EV charging session.

By leveraging a **Stacked Ensemble** (combining RandomForest, XGBoost, CatBoost, and LightGBM) and introducing **probabilistic forecasting**, the model predicts not only the expected energy consumption but also a confidence interval, allowing for better risk management in smart grid operations.

## 🚀 Key Features

- **Hybrid Stacked Ensemble Architecture**: Integrates state-of-the-art gradient boosting algorithms (XGBoost, CatBoost, LightGBM) and Random Forest as base learners.
- **Probabilistic Forecasting**: Predicts energy consumption with a 90% Prediction Interval (PI) using Quantile Regression, offering a median prediction alongside lower and upper bounds.
- **Behavioral Feature Integration**: Incorporates user-centric features like `urgency_score`, `flexibility_index`, `habit_stability`, and `grid_impact_proxy` to enhance model accuracy.
- **Extensive Ablation Studies**: Rigorous comparison between a baseline model and the SOTA ensemble model to quantify the impact of behavioral features.

## 📊 Final Evaluation Results

The models were evaluated primarily on Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score. 

### Stacked Ensemble Model (With Probabilistic Prediction)
The final production model demonstrates exceptional accuracy and highly reliable uncertainty estimation:

- **MAE**: `0.288 kWh`
- **RMSE**: `0.857 kWh`
- **R²**: `0.991`
- **90% PI Coverage**: `97.5%` (The true value falls within the predicted interval 97.5% of the time)
- **Average PI Width**: `4.207 kWh`

*Ablation Study: Removing behavioral features degrades the MAE to 0.299 kWh and RMSE to 0.8945 kWh, proving the value of human-centric features.*

### Baseline Model (XGBoost)
For comparison, a strong XGBoost baseline yielded the following:
- **MAE**: `0.550 kWh`
- **RMSE**: `1.328 kWh`
- **R²**: `0.9802`

---

## 🔍 Example Inference

The probabilistic model outputs detailed predictions including the features used and the confidence interval. 

```text
Session 0:
  Predicted (median): 12.78 kWh
  90% PI: [11.73, 13.37] kWh
  True value: 12.53 kWh
```
*Full probabilistic predictions are saved to `probabilistic_predictions_test.csv`.*

## 📂 Project Structure

- `Code/Stage-2/ev_sota/`: Contains the evolution of the Stacked Ensemble model scripts, ablation study logs, and serialized model artifacts.
- `Code/Stage-2/baseline_model/`: Contains the XGBoost baseline implementation.
- `Code/Inference Results/`: Logs and outputs of the final probabilistic predictions.
- `Code/Final_Metrics/`: Comprehensive logs of evaluation matrices across baseline and ensemble models.
- `Code/Architecture/`: Contains PDF diagrams of the Stacked Ensemble Forecast Model and IEEE-Style Hybrid Architecture.

## 🛠️ Tech Stack & Dependencies

- **Languages**: Python >= 3.12
- **Core ML Libraries**: `xgboost`, `catboost`, `lightgbm`, `scikit-learn`, `torch`
- **Data & Ops**: `pandas`, `numpy`, `optuna` (Hyperparameter tuning), `wandb`, `shap` (Interpretability)
Dependencies are listed in `requirements.txt`.

## 💻 Getting Started

1. Clone the repository.
2. Ensure you have Python >= 3.12. Install dependencies using `uv`:
   ```bash
   uv add -r requirements.txt
   ```
3. Check the `Code/` directory for the various modelling stages and inference scripts.
