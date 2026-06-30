"""Trains both the baseline-from-scratch CNN and the DenseNet-121 transfer
model on the chest X-ray pneumonia dataset, saving both checkpoints so the
eval script can produce a genuine before/after comparison.
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import BASELINE_CNN_PATH, BATCH_SIZE, DENSENET_PATH, MODEL_ARTIFACTS_DIR
from src.data import EVAL_TRANSFORM, TRAIN_TRANSFORM, ChestXrayDataset, load_splits
from src.models.architectures import BaselineCNN, build_densenet121


def train_one_model(model, train_loader, val_loader, epochs: int, lr: float, device) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(
            f"Epoch {epoch + 1}/{epochs}: train_loss={total_loss / len(train_loader):.4f} "
            f"val_acc={correct / total:.4f}"
        )

    return model


def main(epochs: int = 3) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    splits = load_splits()
    train_ds = ChestXrayDataset(splits["train"], TRAIN_TRANSFORM)
    val_ds = ChestXrayDataset(splits["validation"], EVAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Training baseline CNN (from scratch) ===")
    baseline = BaselineCNN()
    baseline = train_one_model(baseline, train_loader, val_loader, epochs, lr=1e-3, device=device)
    torch.save(baseline.state_dict(), BASELINE_CNN_PATH)
    print(f"Saved baseline CNN to {BASELINE_CNN_PATH}")

    print("\n=== Training DenseNet-121 (transfer learning, frozen backbone) ===")
    densenet = build_densenet121(freeze_backbone=True)
    densenet = train_one_model(densenet, train_loader, val_loader, epochs, lr=1e-3, device=device)
    torch.save(densenet.state_dict(), DENSENET_PATH)
    print(f"Saved DenseNet-121 to {DENSENET_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    main(epochs=args.epochs)
