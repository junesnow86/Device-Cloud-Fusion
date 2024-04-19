import pickle

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, squeezenet1_0

from modules.data_utils import SubsetBasedDataset
from modules.distillation import distill
from modules.evaluation import test_accuracy
from modules.models import ImageClassificationModel

# Load the CIFAR10 dataset
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


# Load the models
backbone_out_features = 1000

cloud_model = ImageClassificationModel(
    resnet18(weights=None, num_classes=backbone_out_features),
    backbone_out_features,
    num_classes=10,
)
cloud_model.load_state_dict(
    torch.load("./checkpoints/cloud_resnet18_cifar10_0.4_distilled.pth")
)

device_model = ImageClassificationModel(
    squeezenet1_0(weights=None, num_classes=256),
    256,
    num_classes=10,
)
device_model.load_state_dict(
    torch.load("./checkpoints/device_squeezenet_cifar10_0.2_finetuned.pth")
)


# Evaluate the models before distillation
cloud_accuracy = test_accuracy(cloud_model, test_data)
device_accuracy_before_distillation = test_accuracy(device_model, test_data)

print(f"Cloud model accuracy: {cloud_accuracy:.4f}")
print(f"Device model accuracy: {device_accuracy_before_distillation:.4f}")


# Distill the ensemble to the cloud model
distill(
    device_model,
    cloud_model,
    train_data,
    val_data,
    batch_size=128,
    lr=0.001,
    temperature=20,
    patience=10,
    checkpoint_save_path="./checkpoints/device_squeezenet_cifar10_0.4_second-distilled.pth",
)

device_accuracy_distilled = test_accuracy(device_model, test_data)


# Print the results
print(f"Cloud model accuracy: {cloud_accuracy:.4f}")
print(
    f"Device model accuracy before distillation: {device_accuracy_before_distillation:.4f}"
)
print(f"Device model accuracy after distillation: {device_accuracy_distilled:.4f}")
