import pickle

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.models import (
    mobilenet_v3_small,
    resnet18,
    shufflenet_v2_x1_0,
    squeezenet1_0,
)

from modules.data_utils import SubsetBasedDataset
from modules.distillation import distill
from modules.evaluation import test_accuracy
from modules.models import Ensemble, ImageClassificationModel

# Load the CIFAR10 dataset
# train_transform = v2.Compose(
#     [
#         v2.RandomHorizontalFlip(),
#         v2.RandomVerticalFlip(),
#         v2.ColorJitter(),
#         v2.ToImage(),
#         v2.ToDtype(torch.float32, scale=True),
#         v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#     ]
# )

test_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# with open("checkpoints/data/cloud_subset_cifar10_0.1.pkl", "rb") as f:
#     train_data = pickle.load(f)

# train_ratio = 0.8
# train_size = int(train_ratio * len(train_data))
# val_size = len(train_data) - train_size
# train_data, val_data = random_split(train_data, [train_size, val_size])
# train_data = SubsetBasedDataset(train_data, transform=train_transform)
# val_data = SubsetBasedDataset(val_data, transform=test_transform)
test_data = CIFAR10(root="./data", train=False, transform=test_transform)


# Load the models
backbone_out_features = 1000

cloud_model_before = ImageClassificationModel(
    resnet18(weights=None, num_classes=backbone_out_features),
    backbone_out_features,
    num_classes=10,
)
cloud_model_before.load_state_dict(
    torch.load("./checkpoints/pretrained/cloud_resnet18_cifar10_0.1_pretrained.pth")
)

cloud_model = ImageClassificationModel(
    resnet18(weights=None, num_classes=backbone_out_features),
    backbone_out_features,
    num_classes=10,
)
cloud_model.load_state_dict(
    torch.load("./checkpoints/pretrained/cloud_resnet18_cifar10_0.1_distilled.pth")
)

device_model0 = ImageClassificationModel(
    mobilenet_v3_small(weights=None, num_classes=backbone_out_features),
    backbone_out_features,
    num_classes=10,
)
device_model0.load_state_dict(
    torch.load(
        "./checkpoints/pretrained/device_mobilenet_cifar10_0.1_second-distilled.pth"
    )
)

device_model1 = ImageClassificationModel(
    shufflenet_v2_x1_0(weights=None, num_classes=backbone_out_features),
    backbone_out_features,
    num_classes=10,
)
device_model1.load_state_dict(
    torch.load(
        "./checkpoints/pretrained/device_shufflenet_cifar10_0.1_second-distilled.pth"
    )
)

device_model2 = ImageClassificationModel(
    squeezenet1_0(weights=None, num_classes=256),
    256,
    num_classes=10,
)
device_model2.load_state_dict(
    torch.load(
        "./checkpoints/pretrained/device_squeezenet_cifar10_0.1_second-distilled.pth"
    )
)

# Create the ensemble
device_ensemble = Ensemble([device_model0, device_model1, device_model2])

# Load the fine-tuned device models before second-distillation
device_model0_before = ImageClassificationModel(
    mobilenet_v3_small(weights=None, num_classes=backbone_out_features),
    backbone_out_features,
    num_classes=10,
)
device_model0_before.load_state_dict(
    torch.load("./checkpoints/pretrained/device_mobilenet_cifar10_0.3_finetuned.pth")
)

device_model1_before = ImageClassificationModel(
    shufflenet_v2_x1_0(weights=None, num_classes=backbone_out_features),
    backbone_out_features,
    num_classes=10,
)
device_model1_before.load_state_dict(
    torch.load("./checkpoints/pretrained/device_shufflenet_cifar10_0.3_finetuned.pth")
)

device_model2_before = ImageClassificationModel(
    squeezenet1_0(weights=None, num_classes=256),
    256,
    num_classes=10,
)
device_model2_before.load_state_dict(
    torch.load("./checkpoints/pretrained/device_squeezenet_cifar10_0.3_finetuned.pth")
)

device_ensemble_before = Ensemble(
    [device_model0_before, device_model1_before, device_model2_before]
)


# Evaluate the models before distillation
cloud_accuracy_before = test_accuracy(cloud_model_before, test_data)

cloud_accuracy = test_accuracy(cloud_model, test_data)
device_accuracy0 = test_accuracy(device_model0, test_data)
device_accuracy1 = test_accuracy(device_model1, test_data)
device_accuracy2 = test_accuracy(device_model2, test_data)
device_ensemble_accuracy = test_accuracy(device_ensemble, test_data)

device_accuracy0_before = test_accuracy(device_model0_before, test_data)
device_accuracy1_before = test_accuracy(device_model1_before, test_data)
device_accuracy2_before = test_accuracy(device_model2_before, test_data)
device_ensemble_accuracy_before = test_accuracy(device_ensemble_before, test_data)


# Print the results
print(f"Cloud model accuracy before distillation: {cloud_accuracy_before:.4f}")
print(
    f"Device model 0 accuracy before second-distillation: {device_accuracy0_before:.4f}"
)
print(
    f"Device model 1 accuracy before second-distillation: {device_accuracy1_before:.4f}"
)
print(
    f"Device model 2 accuracy before second-distillation: {device_accuracy2_before:.4f}"
)
print(
    f"Device ensemble accuracy before second-distillation: {device_ensemble_accuracy_before:.4f}"
)
print(f"Cloud model accuracy: {cloud_accuracy:.4f}")
print(f"Device model 0 accuracy: {device_accuracy0:.4f}")
print(f"Device model 1 accuracy: {device_accuracy1:.4f}")
print(f"Device model 2 accuracy: {device_accuracy2:.4f}")
print(f"Device ensemble accuracy: {device_ensemble_accuracy:.4f}")
