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
cd "C:\Manali\Projects\fraud-shield-ai\backend"
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 2) Frontend

```powershell
cd "C:\Manali\Projects\fraud-shield-ai\frontend"
npm install
npm run dev
```

Open the UI at `http://localhost:5173`.

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
