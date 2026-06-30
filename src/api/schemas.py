from pydantic import BaseModel


class PredictResponse(BaseModel):
    predicted_label: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    baseline_model_loaded: bool
    densenet_model_loaded: bool
