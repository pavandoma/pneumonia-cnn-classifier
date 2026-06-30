from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_ARTIFACTS_DIR = PROJECT_ROOT / "src" / "models" / "artifacts"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

BASELINE_CNN_PATH = MODEL_ARTIFACTS_DIR / "baseline_cnn.pt"
DENSENET_PATH = MODEL_ARTIFACTS_DIR / "densenet121_transfer.pt"

# HF dataset used as a laptop-friendly stand-in for the full NIH
# ChestX-ray14 corpus (45GB) -- see README Honest Scope.
DATASET_NAME = "keremberke/chest-xray-classification"
DATASET_CONFIG = "full"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

IMAGE_SIZE = 160
BATCH_SIZE = 16
RANDOM_STATE = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
