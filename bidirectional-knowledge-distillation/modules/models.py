import torch
from scipy import stats


class ImageClassificationModel(torch.nn.Module):
    def __init__(self, backbone, backbone_out_features, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.relu = torch.nn.ReLU(inplace=True)
        self.classifier = torch.nn.Linear(backbone_out_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x


class Ensemble(torch.nn.Module):
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
