import torch
from torchvision.models import resnet18

from modules.data_utils import load_CIFAR10
from modules.evaluation import test_accuracy
from modules.models import ImageClassificationModel
from modules.training import train

train_data, val_data, test_data = load_CIFAR10(root="./data", download=False)

backbone_out_features = 256
backbone = resnet18(weights=None, num_classes=backbone_out_features)
model = ImageClassificationModel(backbone, backbone_out_features, num_classes=10)

train(
    model,
    train_data,
    val_data,
    batch_size=128,
    lr=0.1,
    checkpoint_save_path="./checkpoints/custom_resnet18_cifar10.pth",
)

accuracy = test_accuracy(model, test_data)
print(f"Test accuracy: {accuracy:.4f}")
