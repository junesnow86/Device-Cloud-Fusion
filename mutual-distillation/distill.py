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
cloud_train_data, cloud_test_data = load_Food101(root="./data", download=False)

# Split training data into training and validation
train_ratio = 0.8
original_dataset_length = len(cloud_train_data)
cloud_train_data, cloud_val_data = random_split(
    cloud_train_data,
    [
        int(original_dataset_length * train_ratio),
        original_dataset_length - int(original_dataset_length * train_ratio),
    ],
)

# Load Caltech-101 dataset for participants
participant_train_data, participant_test_data = load_Caltech101(
    root="./data",
    download=False,
    train_ratio=0.9,
)

# Split the training data into 3 parts
start = time.time()
num_participants = 3
participant_train_datas = dirichlet_split(
    participant_train_data, num_participants, num_classes=101
)
print(
    f"Time to split the data among participants with dirichlet: {time.time()-start:.2f}s"
)

# Split training data into training and validation
train_ratio = 0.8
participant_val_datas = []
for train_data in participant_train_datas:
    original_dataset_length = len(train_data)
    train_data, val_data = random_split(
        train_data,
        [
            int(original_dataset_length * train_ratio),
            original_dataset_length - int(original_dataset_length * train_ratio),
        ],
    )
    participant_val_datas.append(val_data)

# Define participants' models and the cloud model
participant_models = []
participant_models.append(squeezenet1_0(weights=None, num_classes=101))
participant_models.append(shufflenet_v2_x1_0(weights=None, num_classes=101))
participant_models.append(mobilenet_v3_small(weights=None, num_classes=101))

cloud_model = resnet18(weights=None, num_classes=101)

# Load models' pre-trained weights
if os.path.exists("./checkpoints/cloud.pth"):
    cloud_model.load_state_dict(torch.load("./checkpoints/cloud_food101.pth"))

for i, model in enumerate(participant_models):
    if os.path.exists(f"./checkpoints/participant_{i+1}.pth"):
        model.load_state_dict(
            torch.load(f"./checkpoints/participant_{i+1}_caltech101.pth")
        )

cloud_pretrained_acc = test(cloud_model, cloud_test_data)
participant_pretrained_accs = []
for i, model in enumerate(participant_models):
    acc = test(model, participant_test_data)
    participant_pretrained_accs.append(acc)

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
            save_path=f"./checkpoints/cloud_kd.pth",
        )
        acc = test(cloud_model, cloud_test_data)
        cloud_kd_accs.append(acc)

participant_kd_accs = []
for i, (model, train_data, val_data) in enumerate(
    zip(participant_models, participant_train_datas, participant_val_datas)
):
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
        train_data,
        val_data,
        batch_size=64,
        save_path=f"./checkpoints/participant_{i+1}_caltech101_kd.pth",
    )
    acc = test(model, participant_test_data)
    participant_kd_accs.append(acc)

print("# Results")
print(f"Cloud pre-trained model accuracy: {cloud_pretrained_acc}")
print(f"Participant pre-trained model accuracies: {participant_pretrained_accs}")
print(f"Cloud KD model accuracy: {cloud_kd_accs}")
print(f"Participant KD model accuracies: {participant_kd_accs}")
