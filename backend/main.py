"""
EV Energy Prediction — FastAPI Backend
Serves probabilistic (conformal) predictions from the trained ensemble stack.

Endpoints
---------
GET /health            — liveness probe
GET /predict/fetch     — random session from the evaluation CSV
GET /predict/run?idx=N — run inference on row N
POST /predict/custom   — run inference on arbitrary user-supplied features
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths — all relative to this file so they work both locally and on Render
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH   = os.path.join(BASE_DIR, "data", "acn_enhanced_final_2019_data.csv")
MODELS_V2   = os.path.join(BASE_DIR, "models", "sota_models_v2")
MODELS_V3   = os.path.join(BASE_DIR, "models", "sota_models_v3")

# Conformal calibration constant (computed offline on calibration set)
Q_HAT = 0.629

# ---------------------------------------------------------------------------
# Feature engineering (mirrors _internal_features.py + predict_api.py logic)
# ---------------------------------------------------------------------------

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df["duration_hours"] = (
        pd.to_datetime(df["disconnectTime"], errors="coerce", utc=True)
        - pd.to_datetime(df["connectionTime"], errors="coerce", utc=True)
    ).dt.total_seconds() / 3600
    df["charging_efficiency"] = df["kWhDelivered"] / df["parsed_kWhRequested"].replace(0, np.nan)
    df["requested_gap"] = df["parsed_kWhRequested"] - df["kWhDelivered"]
    return df


FEATURE_COLS = [
    "parsed_milesRequested", "parsed_WhPerMile", "parsed_minutesAvailable",
    "parsed_kWhRequested", "revisionCount", "hour", "day_of_week_encoded",
    "is_weekend", "is_peak", "urgency_score", "flexibility_index",
    "habit_stability", "grid_impact_proxy", "urgency_flex_interaction",
    "revision_urgency", "habit_grid", "energy_per_minute",
    "request_efficiency", "duration_hours", "charging_efficiency",
    "requested_gap", "hour_sin", "hour_cos",
]
ALL_FEATURE_COLS = FEATURE_COLS + ["station_encoded", "user_encoded"]


def engineer_features(row_df: pd.DataFrame, te_station, te_user) -> pd.DataFrame:
    row_df = row_df.copy()

    row_df["connectionTime"]  = pd.to_datetime(row_df["connectionTime"],  errors="coerce", utc=True)
    row_df["disconnectTime"]  = pd.to_datetime(row_df["disconnectTime"],  errors="coerce", utc=True)

    row_df["hour"]               = row_df["connectionTime"].dt.hour
    row_df["month"]              = row_df["connectionTime"].dt.month
    row_df["day_of_week_encoded"] = pd.factorize(row_df["day_of_week"].fillna("Monday"))[0]
    row_df["is_weekend"]         = (row_df["day_of_week_encoded"] >= 5).astype(int)

    row_df = add_advanced_features(row_df)

    row_df["energy_per_minute"]  = row_df["parsed_kWhRequested"] / (row_df["parsed_minutesAvailable"] + 1e-5)
    row_df["request_efficiency"] = row_df["parsed_milesRequested"] / (row_df["parsed_WhPerMile"] + 1e-5)

    for col, default in [
        ("urgency_score", 50.0),
        ("flexibility_index", 1.0),
        ("grid_impact_proxy", 1.0),
        ("habit_stability", 1.0),
    ]:
        if col not in row_df.columns:
            row_df[col] = default

    row_df["urgency_flex_interaction"] = row_df["urgency_score"] * row_df["flexibility_index"]
    row_df["revision_urgency"]         = row_df.get("revisionCount", 0) * row_df["urgency_score"]
    row_df["habit_grid"]               = row_df["habit_stability"] * row_df["grid_impact_proxy"].fillna(1)
    row_df["is_peak"]                  = row_df["hour"].between(17, 21).astype(int)
    row_df["hour_sin"]                 = np.sin(2 * np.pi * row_df["hour"] / 24)
    row_df["hour_cos"]                 = np.cos(2 * np.pi * row_df["hour"] / 24)

    if "stationID" in row_df.columns:
        row_df["station_encoded"] = te_station.transform(row_df[["stationID"]]).flatten()
    else:
        row_df["station_encoded"] = 0.0

    if "parsed_userID" in row_df.columns:
        row_df["user_encoded"] = te_user.transform(row_df[["parsed_userID"]]).flatten()
    else:
        row_df["user_encoded"] = 0.0

    for col in ALL_FEATURE_COLS:
        if col not in row_df.columns:
            row_df[col] = 0.0

    row_df = row_df.fillna(0)
    return row_df[ALL_FEATURE_COLS]


def run_inference(X: pd.DataFrame, models: dict) -> dict:
    stack = np.column_stack([
        models["rf"].predict(X),
        models["xgb"].predict(X),
        models["cat"].predict(X),
        models["lgb_base"].predict(X),
    ])
    point_pred = models["meta"].predict(stack)[0]
    q05 = models["q05"].predict(X)[0]
    q50 = models["q50"].predict(X)[0]
    q95 = models["q95"].predict(X)[0]

    return {
        "pred_point":        round(float(point_pred), 4),
        "pred_median":       round(float(q50), 4),
        "lower_conformal":   round(float(q05 - Q_HAT), 4),
        "upper_conformal":   round(float(q95 + Q_HAT), 4),
    }


# ---------------------------------------------------------------------------
# App lifecycle — load models once at startup
# ---------------------------------------------------------------------------
_state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Loading models ...")
    _state["te_station"] = joblib.load(os.path.join(MODELS_V2, "te_station.pkl"))
    _state["te_user"]    = joblib.load(os.path.join(MODELS_V2, "te_user.pkl"))
    _state["models"] = {
        "rf":       joblib.load(os.path.join(MODELS_V3, "rf_base.pkl")),
        "xgb":      joblib.load(os.path.join(MODELS_V3, "xgb_base.pkl")),
        "cat":      joblib.load(os.path.join(MODELS_V3, "cat_base.pkl")),
        "lgb_base": joblib.load(os.path.join(MODELS_V3, "lgb_base.pkl")),
        "meta":     lgb.Booster(model_file=os.path.join(MODELS_V3, "meta_lgb.txt")),
        "q05":      lgb.Booster(model_file=os.path.join(MODELS_V2, "lgb_quantile_alpha_5.txt")),
        "q50":      lgb.Booster(model_file=os.path.join(MODELS_V2, "lgb_quantile_alpha_50.txt")),
        "q95":      lgb.Booster(model_file=os.path.join(MODELS_V2, "lgb_quantile_alpha_95.txt")),
    }
    print("[startup] Models loaded -- app ready.")

    # Load the evaluation CSV (needed only for /fetch and /run endpoints)
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df = df.dropna(subset=["kWhDelivered", "parsed_kWhRequested", "parsed_minutesAvailable"])
        _state["df"] = df.reset_index(drop=True)
        print(f"[startup] Dataset loaded: {len(_state['df'])} rows")
    else:
        _state["df"] = None
        print("[startup] Dataset CSV not found -- /fetch and /run endpoints disabled.")

    yield  # app runs

    _state.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="EV Energy Prediction API",
    description="Probabilistic ensemble forecasts for EV charging sessions (conformal intervals).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production to your Vercel/Next.js domain
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": "models" in _state}


@app.get("/predict/fetch")
def fetch_random():
    """Return a random session row from the evaluation dataset."""
    df = _state.get("df")
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not available on this deployment.")

    idx = int(np.random.randint(0, len(df)))
    row = df.iloc[idx]

    return {
        "row_idx":                int(idx),
        "parsed_milesRequested":  float(row.get("parsed_milesRequested", 0)),
        "parsed_WhPerMile":       float(row.get("parsed_WhPerMile", 0)),
        "parsed_minutesAvailable": float(row.get("parsed_minutesAvailable", 0)),
        "parsed_kWhRequested":    float(row.get("parsed_kWhRequested", 0)),
        "revisionCount":          float(row.get("revisionCount", 0)),
        "connectionTime":         str(row.get("connectionTime", "")),
        "disconnectTime":         str(row.get("disconnectTime", "")),
        "day_of_week":            str(row.get("day_of_week", "")),
        "urgency_score":          float(row.get("urgency_score", 50.0)),
        "flexibility_index":      float(row.get("flexibility_index", 1.0)),
        "habit_stability":        float(row.get("habit_stability", 1.0)),
        "grid_impact_proxy":      float(row.get("grid_impact_proxy", 1.0)),
        "true_kWhDelivered":      float(row.get("kWhDelivered", 0)),
    }


@app.get("/predict/run")
def predict_by_index(idx: int = Query(..., description="Row index in the evaluation CSV")):
    """Run model inference on a specific evaluation-set row."""
    df = _state.get("df")
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not available on this deployment.")
    if idx < 0 or idx >= len(df):
        raise HTTPException(status_code=400, detail=f"idx must be in [0, {len(df)-1}]")

    row_df = pd.DataFrame([df.iloc[idx].copy()])
    X = engineer_features(row_df, _state["te_station"], _state["te_user"])
    result = run_inference(X, _state["models"])
    result["true_kWhDelivered"] = float(df.iloc[idx].get("kWhDelivered", 0))
    return result


class SessionInput(BaseModel):
    """Arbitrary session inputs for live prediction (no CSV needed)."""
    parsed_kWhRequested:     float = 20.0
    parsed_minutesAvailable: float = 240.0
    parsed_milesRequested:   float = 0.0
    parsed_WhPerMile:        float = 0.0
    revisionCount:           float = 0.0
    connectionTime:          str   = "2019-06-01T08:00:00+00:00"
    disconnectTime:          str   = "2019-06-01T12:00:00+00:00"
    day_of_week:             str   = "Saturday"
    urgency_score:           float = 50.0
    flexibility_index:       float = 1.0
    habit_stability:         float = 1.0
    grid_impact_proxy:       float = 1.0
    kWhDelivered:            float = 0.0   # used only for feature engineering internals
    stationID:               str   = "unknown"
    parsed_userID:           str   = "unknown"


@app.post("/predict/custom")
def predict_custom(session: SessionInput):
    """Run inference on user-supplied session parameters (no CSV required)."""
    row_df = pd.DataFrame([session.model_dump()])
    X = engineer_features(row_df, _state["te_station"], _state["te_user"])
    return run_inference(X, _state["models"])
