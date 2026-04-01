# FraudShield AI - Credit Card Fraud Detection

Portfolio-grade full-stack project for fraud risk analysis with a polished UI.

## Why this project is impressive

- Full-stack architecture: `React + TypeScript` frontend and `FastAPI` backend.
- Real ML lifecycle: train model from a Hugging Face dataset, evaluate, serve inference API.
- Product-style experience: health checks, one-click training, live metrics dashboard, transaction simulation.

## Stack

- Frontend: React, TypeScript, Vite, Axios
- Backend: FastAPI, scikit-learn, pandas, Hugging Face `datasets`
- Model: Random Forest with class balancing and imputation pipeline

## Features

- Train endpoint (`POST /train`) loads a fraud dataset from Hugging Face and fits the model.
- Metrics panel: ROC-AUC, precision/recall/F1 for fraud class.
- Predict endpoint (`POST /predict`) returns fraud probability + class decision.
- Dynamic UI form generated from model feature names.

## Default dataset

The app uses `David-Egea/Creditcard-fraud-detection` on the Hub by default (same structure as the classic Kaggle creditcard dataset).  
You can type any compatible Hugging Face dataset id in the UI.

Compatibility expectations:
- At least one split (preferably `train`)
- A label column named one of: `Class`, `class`, `label`, `is_fraud`, `fraud`, `target`
- Numeric feature columns

## Run locally

### 1) Backend

```powershell
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

### 2) Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the UI at `http://localhost:5173`.

## Deploy (inference-only, free tier)

Use a **pre-trained model** in `backend/models/` (train once on your PC, then commit `fraud_model.joblib` + `metadata.joblib`). Hosted free tiers are not reliable for long `/train` jobs.

### Backend (example: Render)

1. New **Web Service**, connect the GitHub repo.
2. **Root Directory:** `backend`
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Copy the public URL (e.g. `https://your-api.onrender.com`).

### Frontend (example: Vercel)

1. New project from the same repo.
2. **Root Directory:** `frontend`
3. **Build Command:** `npm run build`
4. **Output Directory:** `dist`
5. **Environment variables:**
   - `VITE_API_URL` = your Render URL (no trailing slash), e.g. `https://your-api.onrender.com`
   - `VITE_INFERENCE_ONLY` = `true` (hides training UI; loads features on load)

Copy `frontend/.env.example` to `frontend/.env.production` locally only if you want to test a production build; on Vercel set the same vars in the dashboard.

### CORS

The API allows all origins in dev. For a stricter production setup, restrict `allow_origins` in `backend/app/main.py` to your Vercel domain.

## API quick check

- `GET /health`
- `POST /train`
- `GET /features`
- `POST /predict`

## Roadmap ideas

- Add XGBoost/LightGBM and model comparison tab
- Add SHAP explanations per prediction
- Add transaction history and analyst notes
- Add auth + role-based dashboard (analyst/admin)
