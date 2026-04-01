from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .ml import load_artifacts, predict_one, train_model
from .schemas import FeatureInfoResponse, HealthResponse, PredictRequest, PredictResponse, TrainRequest


app = FastAPI(title="FraudShield AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    model, _ = load_artifacts()
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/train")
def train(payload: TrainRequest):
    try:
        result = train_model(
            dataset_id=payload.dataset_id,
            test_size=payload.test_size,
            random_state=payload.random_state,
        )
        return {"ok": True, "metrics": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/features", response_model=FeatureInfoResponse)
def features():
    _, metadata = load_artifacts()
    if not metadata:
        raise HTTPException(status_code=404, detail="Model not trained yet.")
    return {"feature_names": metadata["feature_names"]}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    model, metadata = load_artifacts()
    if model is None or metadata is None:
        raise HTTPException(status_code=404, detail="Model not trained yet. Call /train first.")
    return predict_one(model, metadata, payload.features)

