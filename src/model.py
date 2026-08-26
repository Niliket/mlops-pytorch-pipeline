import torch
import torch.nn as nn
from torchvision import models

def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        # Modify the first conv layer and fc layer for CIFAR-10 (32x32 images)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        # Default lightweight CNN fallback
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
