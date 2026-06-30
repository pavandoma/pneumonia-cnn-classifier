"""Two model architectures, matching the resume's comparison framing:

- BaselineCNN: a small CNN trained from scratch (no pretraining) -- the
  "baseline CNN" the transfer-learning model is measured against.
- build_densenet121(): ImageNet-pretrained DenseNet-121 with its classifier
  head replaced and fine-tuned -- the resume's primary architecture.
- build_inception_v3(): also implemented and runnable (matches the
  resume's "DenseNet-121 / Inception-V3" framing), but this rebuild's
  reported numbers come from DenseNet-121 only, due to CPU training time
  constraints -- see README Honest Scope.
"""

import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_densenet121(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def build_inception_v3(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    model.aux_logits = False
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
