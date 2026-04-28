# EV Energy Prediction — FastAPI Backend

A lightweight FastAPI service that wraps the trained probabilistic ensemble models.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/predict/fetch` | Random session from evaluation CSV |
| `GET` | `/predict/run?idx=N` | Inference on row N of the CSV |
| `POST` | `/predict/custom` | Inference on arbitrary session inputs |

## Local development

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Interactive docs → http://localhost:8000/docs

## Required model & data files

The service expects the following layout inside `backend/`:

```
backend/
├── main.py
├── requirements.txt
├── Procfile
├── data/
│   └── acn_enhanced_final_2019_data.csv     ← copy from Code/Stage-2/ev_sota/
└── models/
    ├── sota_models_v2/
    │   ├── te_station.pkl
    │   ├── te_user.pkl
    │   ├── lgb_quantile_alpha_5.txt
    │   ├── lgb_quantile_alpha_50.txt
    │   └── lgb_quantile_alpha_95.txt
    └── sota_models_v3/
        ├── rf_base.pkl
        ├── xgb_base.pkl
        ├── cat_base.pkl
        ├── lgb_base.pkl
        └── meta_lgb.txt
```

Run the helper script to copy everything in one go:

```bash
python backend/setup_assets.py
```

## Deploy to Render

1. Push this repo to GitHub.
2. Go to https://render.com → **New → Web Service**.
3. Connect the repo; Render auto-detects `render.yaml` and pre-fills all settings.
4. Click **Create Web Service** — done.
5. Your live URL will be `https://ev-energy-prediction-api.onrender.com`.

After deploying, set `NEXT_PUBLIC_API_URL` in your Next.js project to that URL.
