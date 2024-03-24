import os

import torch
import torchvision.transforms as transforms
from data_utils import MappedCaltech101, dirichlet_split
from pipeline import distill, test, train
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100, Caltech101
from torchvision.models import (
    mobilenet_v3_small,
    resnet18,
    shufflenet_v2_x1_0,
    squeezenet1_0,
)

# Load Caltech-101 dataset for cloud training
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

cloud_dataset = MappedCaltech101(
    offset=10,
    root="./data",
    download=False,
    transform=transform,
    target_type="category",
)
labels_count = {}
for _, label in cloud_dataset:
    if label not in labels_count:
        labels_count[label] = 0
    labels_count[label] += 1
print(labels_count)
cloud_train_data, cloud_test_data = random_split(
    cloud_dataset,
    [int(len(cloud_dataset) * 0.8), len(cloud_dataset) - int(len(cloud_dataset) * 0.8)],
)
cloud_train_data, cloud_val_data = random_split(
    cloud_train_data,
    [
        int(len(cloud_train_data) * 0.8),
        len(cloud_train_data) - int(len(cloud_train_data) * 0.8),
    ],
)

# Load CIFAR-100 dataset for participants training
transform = transforms.Compose(
    [
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

participant_train_data = CIFAR10(
    root="./data", download=False, transform=transform, train=True
)
participant_test_data = CIFAR10(
    root="./data", download=False, transform=transform, train=False
)
participant_val_data, participant_test_data = random_split(
    participant_test_data,
    [
        int(len(participant_test_data) * 0.5),
        len(participant_test_data) - int(len(participant_test_data) * 0.5),
    ],
)

# Define participants' models and the cloud model
num_participants = 3
participant_train_datas = dirichlet_split(participant_train_data, num_participants)
participant_models = []
participant_models.append(squeezenet1_0(weights=None, num_classes=111))
participant_models.append(shufflenet_v2_x1_0(weights=None, num_classes=111))
participant_models.append(mobilenet_v3_small(weights=None, num_classes=111))

cloud_model = resnet18(weights=None, num_classes=111)

# Train models
participant_accs = []
for i, (model, train_data) in enumerate(
    zip(participant_models, participant_train_datas)
):
    if os.path.exists(f"./checkpoints/participant_{i+1}_cifar10.pth"):
        model.load_state_dict(
            torch.load(f"./checkpoints/participant_{i+1}_cifar10.pth")
        )
        acc = test(model, participant_test_data)
        participant_accs.append(acc)
        continue
    print(f">>> Training participant {i+1}")
    train(
        model,
        train_data,
        participant_val_data,
        batch_size=64,
        lr=0.0001,
        save_path=f"./checkpoints/participant_{i+1}_cifar10.pth",
    )
    acc = test(model, participant_test_data)
    participant_accs.append(acc)

if os.path.exists("./checkpoints/cloud_caltech101.pth"):
    cloud_model.load_state_dict(torch.load("./checkpoints/cloud_caltech101.pth"))
    cloud_acc = test(cloud_model, cloud_test_data)
else:
    print(">>> Training the cloud model...")
    train(
        cloud_model,
        cloud_train_data,
        cloud_val_data,
        batch_size=32,
        save_path="./checkpoints/cloud_caltech101.pth",
    )
    cloud_acc = test(cloud_model, cloud_test_data)

# Mutual Distillation
cloud_accs_kd = []
if os.path.exists("./checkpoints/cloud_kd.pth"):
    cloud_model.load_state_dict(torch.load("./checkpoints/cloud_kd.pth"))
    acc = test(cloud_model, cloud_test_data)
    cloud_accs_kd.append(acc)
else:
    for i, participant in enumerate(participant_models):
        print(f">>> Distilling participant {i+1}")
        distill(
            participant,
            cloud_model,
            cloud_train_data,
            cloud_val_data,
            batch_size=16,
            save_path=f"./checkpoints/cloud_kd.pth",
        )
        acc = test(cloud_model, cloud_test_data)
        cloud_accs_kd.append(acc)

participant_accs_kd = []
for i, (model, train_data) in enumerate(
    zip(participant_models, participant_train_datas)
):
    if os.path.exists(f"./checkpoints/participant_{i+1}_cifar10_kd.pth"):
        model.load_state_dict(
            torch.load(f"./checkpoints/participant_{i+1}_cifar10_kd.pth")
        )
        acc = test(model, participant_test_data)
        participant_accs_kd.append(acc)
        continue
    print(f">>> Distilling the cloud model to participant {i+1}")
    train(
        model,
        train_data,
        participant_val_data,
        batch_size=16,
        save_path=f"./checkpoints/participant_{i+1}_cifar10_kd.pth",
    )
    acc = test(model, participant_test_data)
    participant_accs_kd.append(acc)

print(f"Cloud model accuracy: {cloud_acc}")
print(f"Participant models accuracy: {participant_accs}")
print(f"Cloud accuracy after KD: {cloud_accs_kd}")
print(f"Participant accuracy after KD: {participant_accs_kd}")
