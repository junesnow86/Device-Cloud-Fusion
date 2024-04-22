import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1

from modules.evaluation import test_accuracy
from modules.training import train

model = nn.Sequential(
    squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT, num_classes=1000),
    nn.Linear(1000, 100),
)


# Load CIFAR-100 dataset
transform = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

train_data = CIFAR10(root="./data", train=True, transform=transform)
test_data = CIFAR10(root="./data", train=False, transform=transform)

train_data_size = int(0.9 * len(train_data))
val_data_size = len(train_data) - train_data_size
train_data, val_data = random_split(train_data, [train_data_size, val_data_size])

train(
    model,
    train_data,
    val_data,
    batch_size=128,
    lr=0.01,
    checkpoint_save_path="./checkpoints/squeezenet_1_1_cifar10.pth",
)

accuracy = test_accuracy(model, test_data)
print(f"Test accuracy: {accuracy:.4f}")
