"""Inference wrappers for both models, used by the API and eval scripts."""

import torch
from PIL import Image

from src.config import BASELINE_CNN_PATH, CLASS_NAMES, DENSENET_PATH
from src.data import EVAL_TRANSFORM
from src.models.architectures import BaselineCNN, build_densenet121

_baseline_model = None
_densenet_model = None


def _load_baseline():
    global _baseline_model
    if _baseline_model is None:
        model = BaselineCNN()
        model.load_state_dict(torch.load(BASELINE_CNN_PATH, map_location="cpu"))
        model.eval()
        _baseline_model = model
    return _baseline_model


def _load_densenet():
    global _densenet_model
    if _densenet_model is None:
        model = build_densenet121(freeze_backbone=True)
        model.load_state_dict(torch.load(DENSENET_PATH, map_location="cpu"))
        model.eval()
        _densenet_model = model
    return _densenet_model


def _predict(model, image: Image.Image) -> tuple[str, float]:
    tensor = EVAL_TRANSFORM(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = int(torch.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx])


def predict_baseline(image: Image.Image) -> tuple[str, float]:
    return _predict(_load_baseline(), image)


def predict_densenet(image: Image.Image) -> tuple[str, float]:
    return _predict(_load_densenet(), image)
