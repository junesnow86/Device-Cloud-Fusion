import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)

from modules.evaluation import test_accuracy
from modules.training import train

transform = T.Compose(
    [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_data = CIFAR10(root="./data", train=True, transform=transform)
test_data = CIFAR10(root="./data", train=False, transform=transform)

cloud_ratio = 0.3
cloud_size = int(cloud_ratio * len(train_data))
train_data_cloud, train_data_device = random_split(
    train_data, [cloud_size, len(train_data) - cloud_size]
)

val_ratio = 0.1

val_size = int(val_ratio * len(train_data_device))
train_size = len(train_data_device) - val_size
train_data_device, val_data_device = random_split(
    train_data_device, [train_size, val_size]
)

val_size = int(val_ratio * len(train_data_cloud))
train_size = len(train_data_cloud) - val_size
train_data_cloud, val_data_cloud = random_split(
    train_data_cloud, [train_size, val_size]
)


model_cloud = resnet18(weights=ResNet18_Weights.DEFAULT)
model_cloud.fc = nn.Linear(model_cloud.fc.in_features, 10)

model_device = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
model_device.classifier[3] = nn.Linear(model_device.classifier[3].in_features, 10)


train(
    model_cloud,
    train_data_cloud,
    val_data_cloud,
    lr=0.01,
    checkpoint_save_path="cloud_model.pth",
)
train(
    model_device,
    train_data_device,
    val_data_device,
    lr=0.01,
    checkpoint_save_path="device_model.pth",
)

accuracy_cloud = test_accuracy(model_cloud, test_data)
accuracy_device = test_accuracy(model_device, test_data)

print(f"Accuracy of cloud model: {accuracy_cloud}")
print(f"Accuracy of device model: {accuracy_device}")
