import pickle

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small, resnet18

from modules.data_utils import SubsetBasedDataset
from modules.distillation import distill
from modules.evaluation import test_accuracy
from modules.models import ImageClassificationModel

cloud_backbone = resnet18(weights=None, num_classes=1000)
cloud_model = ImageClassificationModel(cloud_backbone, 1000, num_classes=10)
cloud_model.load_state_dict(
    torch.load("checkpoints/pretrained/cloud_resnet18_cifar10_0.1_pretrained.pth")
)

device_backbone = mobilenet_v3_small(
    weights=MobileNet_V3_Small_Weights.DEFAULT, num_classes=1000
)
device_model = ImageClassificationModel(device_backbone, 1000, num_classes=10)


train_transform = v2.Compose(
    [
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ColorJitter(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

test_transform = v2.Compose(
    [
        # v2.RandomHorizontalFlip(),
        # v2.RandomVerticalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

with open("checkpoints/data/cloud_subset_cifar10_0.1.pkl", "rb") as f:
    train_data = pickle.load(f)

train_ratio = 0.8
train_size = int(train_ratio * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = random_split(train_data, [train_size, val_size])
train_data = SubsetBasedDataset(train_data, transform=train_transform)
val_data = SubsetBasedDataset(val_data, transform=test_transform)
test_data = CIFAR10(root="./data", train=False, transform=test_transform)

teacher_accuracy = test_accuracy(cloud_model, test_data)
print(f"Teacher accuracy: {teacher_accuracy:.4f}")

distill(
    device_model,
    cloud_model,
    train_data,
    val_data,
    batch_size=128,
    lr=0.01,
    checkpoint_save_path="./checkpoints/pretrained/device_mobilenet_cifar10_0.1_distilled.pth",
)

accuracy = test_accuracy(device_model, test_data)
print(f"Teacher accuracy: {teacher_accuracy:.4f}")
print(f"Test accuracy: {accuracy:.4f}")
