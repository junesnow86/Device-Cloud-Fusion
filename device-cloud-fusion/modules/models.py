import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


class ImageClassificationModel(nn.Module):
    def __init__(self, backbone, backbone_out_features, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(backbone_out_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x


class Ensemble(nn.Module):
    def __init__(self, models, mode="mean"):
        super(Ensemble, self).__init__()
        self.models = models
        assert mode in ["mean", "voting"]
        self.mode = mode

    def forward(self, x):
        if self.mode == "mean":
            outputs = torch.stack([model(x) for model in self.models])
            return torch.mean(outputs, dim=0)
        elif self.mode == "voting":
            outputs = torch.stack([model(x).argmax(dim=1) for model in self.models])
            outputs = outputs.detach().cpu().numpy()
            voted_outputs = stats.mode(outputs, axis=0)[0]
            return torch.from_numpy(voted_outputs).to(x.device)

    def to(self, device):
        self.models = [model.to(device) for model in self.models]


class SimpleCNN(nn.Module):
    def __init__(self, in_channels, input_size, num_classes):
        super().__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear((input_size // 4) ** 2 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LowLevelEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class FusionModel(nn.Module):
    def __init__(self, cloud_part, control_part, shared_encoder):
        super().__init__()
        self.cloud_part = cloud_part
        self.control_part = control_part
        self.shared_encoder = shared_encoder

    def forward(self, x):
        x = self.shared_encoder(x)
        cloud_logits = self.cloud_part(x)
        control_logits = self.control_part(x)
        outputs = cloud_logits + control_logits
        return outputs
