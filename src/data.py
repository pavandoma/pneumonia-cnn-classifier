"""Loads the chest X-ray pneumonia dataset and wraps it as a PyTorch
Dataset with torchvision transforms (resize, normalization, and -- for the
training split only -- light augmentation: random flip and rotation).
"""

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import DATASET_CONFIG, DATASET_NAME, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class ChestXrayDataset(Dataset):
    def __init__(self, hf_split, transform):
        self.hf_split = hf_split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx):
        row = self.hf_split[idx]
        image = row["image"].convert("RGB")
        label = row["labels"]
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def load_splits():
    return load_dataset(DATASET_NAME, name=DATASET_CONFIG)
