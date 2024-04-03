import os
import pickle
import time

import torch
from data_utils import dirichlet_split, load_Caltech101, load_Food101
from pipeline import distill, test
from torch.utils.data import random_split
from torchvision.models import (
    mobilenet_v3_small,
    resnet18,
    shufflenet_v2_x1_0,
    squeezenet1_0,
)

print("# Experiment Information")
print("cloud dataset: Food-101")
print("participant dataset: Caltech-101")
print("participant number: 3")
print("participant data split: Dirichlet")
print("cloud model: ResNet-18")
print("participant models: SqueezeNet, ShuffleNet, MobileNetV3")

# Load Food-101 dataset for cloud
with open("./checkpoints/data/cloud_train_data.pkl", "rb") as f:
    cloud_train_data = pickle.load(f)
with open("./checkpoints/data/cloud_val_data.pkl", "rb") as f:
    cloud_val_data = pickle.load(f)
with open("./checkpoints/data/cloud_test_data.pkl", "rb") as f:
    cloud_test_data = pickle.load(f)

# Load Caltech-101 dataset for participants
with open("./checkpoints/data/participant_test_data.pkl", "rb") as f:
    participant_test_data = pickle.load(f)

# Split training data into training and validation
participant_train_datas = []
participant_val_datas = []
for i in range(3):
    with open(f"./checkpoints/data/participant_{i+1}_train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(f"./checkpoints/data/participant_{i+1}_val_data.pkl", "rb") as f:
        val_data = pickle.load(f)
    participant_train_datas.append(train_data)
    participant_val_datas.append(val_data)

# Define participants' models and the cloud model
participant_models = []
participant_models.append(squeezenet1_0(weights=None, num_classes=101))
participant_models.append(shufflenet_v2_x1_0(weights=None, num_classes=101))
participant_models.append(mobilenet_v3_small(weights=None, num_classes=101))

cloud_model = resnet18(weights=None, num_classes=101)

# Load models' pre-trained weights
if os.path.exists("./checkpoints/cloud_food101.pth"):
    cloud_model.load_state_dict(torch.load("./checkpoints/cloud_food101.pth"))

for i, model in enumerate(participant_models):
    if os.path.exists(f"./checkpoints/participant_{i+1}_caltech101.pth"):
        model.load_state_dict(
            torch.load(f"./checkpoints/participant_{i+1}_caltech101.pth")
        )
        for param in model.parameters():
            param.requires_grad = True

cloud_pretrained_acc = test(cloud_model, cloud_test_data)
participant_pretrained_accs = []
participant_pretrained_local_accs = []
for i, model in enumerate(participant_models):
    acc = test(model, participant_test_data)
    participant_pretrained_accs.append(acc)
    local_acc = test(model, participant_val_datas[i])
    participant_pretrained_local_accs.append(local_acc)

# Mutual Distillation
cloud_kd_accs = []
if os.path.exists("./checkpoints/cloud_kd.pth"):
    cloud_model.load_state_dict(torch.load("./checkpoints/cloud_kd.pth"))
    acc = test(cloud_model, cloud_test_data)
    cloud_kd_accs.append(acc)
else:
    for i, participant in enumerate(participant_models):
        print(f">>> Distilling participant {i+1}")
        distill(
            participant,
            cloud_model,
            cloud_train_data,
            cloud_val_data,
            batch_size=64,
            lr=0.0005,
            save_path=f"./checkpoints/cloud_kd.pth",
        )
        acc = test(cloud_model, cloud_test_data)
        cloud_kd_accs.append(acc)

participant_kd_accs = []
participant_kd_local_accs = []
for i, model in enumerate(participant_models):
    if os.path.exists(f"./checkpoints/participant_{i+1}_caltech101_kd.pth"):
        model.load_state_dict(
            torch.load(f"./checkpoints/participant_{i+1}_caltech101_kd.pth")
        )
        acc = test(model, participant_test_data)
        participant_kd_accs.append(acc)
        continue

    print(f">>> Distilling the cloud model to participant {i+1}")
    distill(
        cloud_model,
        model,
        cloud_train_data,
        cloud_val_data,
        batch_size=64,
        lr=0.0005,
        save_path=f"./checkpoints/participant_{i+1}_caltech101_kd.pth",
    )
    acc = test(model, participant_test_data)
    participant_kd_accs.append(acc)
    local_acc = test(model, val_data)
    participant_kd_local_accs.append(local_acc)

print("# Results")
print(f"Cloud pre-trained model accuracy: {cloud_pretrained_acc}")
print(
    f"Participant pre-trained model local accuracies: {participant_pretrained_local_accs}"
)
print(f"Participant pre-trained model accuracies: {participant_pretrained_accs}")
print(f"Cloud KD model accuracy: {cloud_kd_accs}")
print(f"Participant KD model local accuracies: {participant_kd_local_accs}")
print(f"Participant KD model accuracies: {participant_kd_accs}")
