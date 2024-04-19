import pickle

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18

from modules.data_utils import SubsetBasedDataset
from modules.evaluation import test_accuracy
from modules.models import ImageClassificationModel
from modules.training import train

backbone = resnet18(weights=ResNet18_Weights.DEFAULT, num_classes=1000)
model = ImageClassificationModel(backbone, 1000, num_classes=10)


train_transform = v2.Compose(
    [
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(32, padding=4),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

test_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

with open("data/cloud_subset_cifar10_0.4.pkl", "rb") as f:
    train_data = pickle.load(f)

train_ratio = 0.9
train_size = int(train_ratio * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = random_split(train_data, [train_size, val_size])
train_data = SubsetBasedDataset(train_data, transform=train_transform)
val_data = SubsetBasedDataset(val_data, transform=test_transform)
test_data = CIFAR10(root="./data", train=False, transform=test_transform)


train(
    model,
    train_data,
    val_data,
    batch_size=128,
    lr=0.01,
    checkpoint_save_path="./checkpoints/cloud_resnet18_cifar10_0.4_pretrained.pth",
)

accuracy = test_accuracy(model, test_data)
print(f"Test accuracy: {accuracy:.4f}")
