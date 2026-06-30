import io

from fastapi import FastAPI, UploadFile
from PIL import Image

from src.api.schemas import HealthResponse, PredictResponse
from src.config import BASELINE_CNN_PATH, DENSENET_PATH
from src.inference import predict_baseline, predict_densenet

app = FastAPI(title="Pneumonia Chest X-ray Classifier")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        baseline_model_loaded=BASELINE_CNN_PATH.exists(),
        densenet_model_loaded=DENSENET_PATH.exists(),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile, use_densenet: bool = True) -> PredictResponse:
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    if use_densenet:
        label, confidence = predict_densenet(image)
    else:
        label, confidence = predict_baseline(image)

    return PredictResponse(predicted_label=label, confidence=confidence)
