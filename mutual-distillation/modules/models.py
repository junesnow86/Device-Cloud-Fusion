import torch


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
