import torch
from torchvision.models import resnet18

from modules.data_utils import load_CIFAR10
from modules.evaluation import test_accuracy
from modules.training import train

train_data, val_data, test_data = load_CIFAR10(root="./data", download=False)

model = resnet18(weights=None, num_classes=10)

train(
    model,
    train_data,
    val_data,
    batch_size=128,
    lr=0.1,
    # checkpoint_save_path="./checkpoints/resnet18_cifar10.pth",
)

accuracy = test_accuracy(model, test_data)
print(f"Test accuracy: {accuracy:.4f}")

backbone = resnet18(weights=None, num_classes=256)
classifier = torch.nn.Linear(256, 10)
new_model = torch.nn.Sequential(backbone, torch.nn.ReLU(), classifier)

train(
    new_model,
    train_data,
    val_data,
    batch_size=128,
    lr=0.1,
    # checkpoint_save_path="./checkpoints/resnet18_cifar10.pth",
)

accuracy = test_accuracy(new_model, test_data)
print(f"Test accuracy with extra classifier: {accuracy:.4f}")
