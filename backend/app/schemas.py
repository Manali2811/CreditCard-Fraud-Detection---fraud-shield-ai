from typing import Dict, List

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    dataset_id: str = Field(
        default="David-Egea/Creditcard-fraud-detection",
        description="Hugging Face dataset id",
    )
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    random_state: int = Field(default=42)


class PredictRequest(BaseModel):
    features: Dict[str, float]


class PredictResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    threshold: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class FeatureInfoResponse(BaseModel):
    feature_names: List[str]

