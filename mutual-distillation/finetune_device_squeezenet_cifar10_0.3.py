import pickle

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.models import squeezenet1_0

from modules.data_utils import SubsetBasedDataset
from modules.evaluation import test_accuracy
from modules.models import ImageClassificationModel
from modules.training import train

backbone_out_features = 256
backbone = squeezenet1_0(weights=None, num_classes=backbone_out_features)
model = ImageClassificationModel(backbone, backbone_out_features, num_classes=10)
model.load_state_dict(
    torch.load("./checkpoints/device_squeezenet_cifar10_0.1_distilled.pth")
)


train_transform = v2.Compose(
    [
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
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

with open("checkpoints/data/remaining_subset2_cifar10_0.3.pkl", "rb") as f:
    train_data = pickle.load(f)

train_ratio = 0.8
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
    checkpoint_save_path="./checkpoints/device_squeezenet_cifar10_0.3_finetuned.pth",
)

accuracy = test_accuracy(model, test_data)
print(f"Test accuracy: {accuracy:.4f}")
