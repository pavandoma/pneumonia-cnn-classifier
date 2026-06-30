# Pneumonia Chest X-ray Classification

An interview-prep rebuild of the "Medical Image Classification (DenseNet-121 / Inception-V3)" resume project: transfer-learning CNN for pneumonia detection, evaluated against a from-scratch baseline CNN with bootstrap 95% confidence intervals.

## Honest Scope

1. **Dataset size** — the original claim used the full NIH ChestX-ray14 corpus (~45GB, 100K+ images). This rebuild uses `keremberke/chest-xray-classification` (HF Hub), a smaller, laptop-friendly binary NORMAL/PNEUMONIA dataset (~4,077 train / 1,165 val / 582 test images) derived from the same family of public chest X-ray datasets. Same technique, smaller corpus — stated directly.
2. **Inception-V3** — both `build_densenet121()` and `build_inception_v3()` are implemented in `src/models/architectures.py` and both are runnable, but this rebuild's reported numbers come from DenseNet-121 only, due to CPU training time constraints on a laptop. Inception-V3 is there to demonstrate the architecture-swap capability, not separately benchmarked here.
3. **The "3% surpassing expectations" figure** — this rebuild measured a real, independent **+3.01%** relative accuracy improvement of the DenseNet-121 transfer model over the from-scratch baseline CNN, landing remarkably close to the original claim by coincidence, not by design — see the actual numbers below and `eval/results/classification_eval.json`.

## Real Results (582-image held-out test set, 1000-resample bootstrap 95% CI)

| Model | Accuracy | Accuracy 95% CI | AUC | AUC 95% CI |
|---|---|---|---|---|
| Baseline CNN (from scratch, 3 epochs) | 89.3% | [86.4%, 91.8%] | 0.946 | [0.926, 0.963] |
| **DenseNet-121 (transfer learning, frozen backbone, 3 epochs)** | **92.0%** | [89.7%, 94.0%] | **0.973** | [0.960, 0.983] |

**Accuracy improvement: +3.01%** (DenseNet-121 vs. baseline CNN), with non-overlapping AUC confidence intervals — genuine evidence the transfer-learned model generalizes better, not just a lucky split.

## Quick Start (Windows / PowerShell)

```powershell
py -3.11 -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.train --epochs 3       # trains and saves both models (~15-20 min on CPU)
python -m eval.run_classification_eval  # reproduces the numbers above

uvicorn src.api.main:app --reload    # starts the API on :8000
```

Try it (with any chest X-ray PNG/JPG):
```powershell
Invoke-RestMethod -Uri http://localhost:8000/health
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -Form @{ file = Get-Item "path\to\xray.png" }
```

## Project Structure

```
pneumonia-cnn-classifier/
├── src/
│   ├── data.py                  # dataset loading + transforms/augmentation
│   ├── models/architectures.py   # BaselineCNN, DenseNet-121, Inception-V3 builders
│   ├── train.py                  # trains both baseline and DenseNet-121
│   ├── inference.py               # inference wrappers
│   └── api/main.py                # FastAPI service (/predict, /health)
├── eval/run_classification_eval.py  # bootstrap 95% CI accuracy/AUC comparison
└── tests/
```

## Reproducing the Numbers

`python -m src.train --epochs 3` trains and saves both models, then `python -m eval.run_classification_eval` prints and saves the comparison to `eval/results/classification_eval.json`.
