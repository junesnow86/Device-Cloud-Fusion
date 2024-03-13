import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding="same", stride=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class Classifier(nn.Module):
    def __init__(self, in_features, num_classes) -> None:
        super().__init__()
        self.activation = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.activation(x)
        x = self.fc(x)
        return x
