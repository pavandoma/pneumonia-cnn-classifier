import io
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image


def _build_client() -> TestClient:
    from src.api.main import app

    return TestClient(app)


def test_health_endpoint():
    client = _build_client()
    response = client.get("/health")
    assert response.status_code == 200


def _fake_image_bytes() -> bytes:
    image = Image.new("RGB", (160, 160), color=(128, 128, 128))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@patch("src.api.main.predict_densenet", return_value=("PNEUMONIA", 0.92))
def test_predict_endpoint(mock_predict):
    client = _build_client()
    response = client.post(
        "/predict",
        files={"file": ("test.png", _fake_image_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] == "PNEUMONIA"
    assert body["confidence"] == 0.92
