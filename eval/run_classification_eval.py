"""Evaluates the baseline-from-scratch CNN vs. the DenseNet-121 transfer
model on the held-out test split, reporting accuracy/AUC with a bootstrap
95% confidence interval -- the direct measurable analog of "evaluated
pneumonia detection model performance using a 95% confidence interval for
validation metrics, confirming generalization gains against baseline CNN
models."
"""

import json
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from src.config import EVAL_RESULTS_DIR
from src.data import EVAL_TRANSFORM, ChestXrayDataset, load_splits
from src.inference import _load_baseline, _load_densenet


def get_predictions(model, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.softmax(model(images), dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def bootstrap_ci(preds, probs, labels, n_bootstrap: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(labels)
    acc_samples, auc_samples = [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        acc_samples.append(accuracy_score(labels[idx], preds[idx]))
        if len(set(labels[idx])) > 1:
            auc_samples.append(roc_auc_score(labels[idx], probs[idx]))

    return {
        "accuracy_mean": float(np.mean(acc_samples)),
        "accuracy_95ci": [float(np.percentile(acc_samples, 2.5)), float(np.percentile(acc_samples, 97.5))],
        "auc_mean": float(np.mean(auc_samples)),
        "auc_95ci": [float(np.percentile(auc_samples, 2.5)), float(np.percentile(auc_samples, 97.5))],
    }


def main() -> None:
    splits = load_splits()
    test_ds = ChestXrayDataset(splits["test"], EVAL_TRANSFORM)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    baseline = _load_baseline()
    densenet = _load_densenet()

    baseline_preds, baseline_probs, labels = get_predictions(baseline, test_loader)
    densenet_preds, densenet_probs, _ = get_predictions(densenet, test_loader)

    baseline_stats = bootstrap_ci(baseline_preds, baseline_probs, labels)
    densenet_stats = bootstrap_ci(densenet_preds, densenet_probs, labels)

    accuracy_improvement_pct = (
        (densenet_stats["accuracy_mean"] - baseline_stats["accuracy_mean"])
        / baseline_stats["accuracy_mean"]
        * 100
    )

    summary = {
        "n_test_samples": len(labels),
        "baseline_cnn_from_scratch": baseline_stats,
        "densenet121_transfer_learning": densenet_stats,
        "accuracy_improvement_pct": accuracy_improvement_pct,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": (
            "95% CIs computed via 1000-resample bootstrap over the test set, "
            "matching the resume's '95% confidence interval for validation metrics' "
            "claim with an actual, reproducible bootstrap procedure."
        ),
    }

    print(json.dumps(summary, indent=2))

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (EVAL_RESULTS_DIR / "classification_eval.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
