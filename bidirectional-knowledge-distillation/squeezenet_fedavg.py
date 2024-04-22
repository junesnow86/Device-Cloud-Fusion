import pickle

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.models import squeezenet1_0

from modules.data_utils import SubsetBasedDataset
from modules.functional import test_accuracy, train
from modules.models import ImageClassificationModel


def fed_avg(model_weights, sample_numbers):
    avg_weights = {}
    for key in model_weights[0].keys():
        layer_weights = [
            model_weight[key].clone().detach() * num
            for model_weight, num in zip(model_weights, sample_numbers)
        ]
        layer_weights_avg = sum(layer_weights) / sum(sample_numbers)
        avg_weights[key] = layer_weights_avg

    return avg_weights


global_model = ImageClassificationModel(
    squeezenet1_0(weights=None, num_classes=256), 256, num_classes=10
)
global_model.load_state_dict(
    torch.load("./checkpoints/device_squeezenet_cifar10_0.4_distilled.pth")
)

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

train_datas, val_datas = [], []

train_ratio = 0.9

for i in range(3):
    with open(f"data/remaining_subset{i}_cifar10_0.2.pkl", "rb") as f:
        train_data = pickle.load(f)

    train_size = int(train_ratio * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    train_datas.append(SubsetBasedDataset(train_data, transform=train_transform))
    val_datas.append(SubsetBasedDataset(val_data, transform=test_transform))

test_data = CIFAR10(root="./data", train=False, transform=test_transform)

accuracy = test_accuracy(global_model, test_data)
print(f"Round 0: {accuracy}")

accuracy_list = []
accuracy_list.append(accuracy)

num_rounds = 20
best_accuracy = 0.0
for round_num in range(num_rounds):
    model_weights = []
    for i in range(3):
        device_model = ImageClassificationModel(
            squeezenet1_0(weights=None, num_classes=256), 256, num_classes=10
        )
        device_model.load_state_dict(global_model.state_dict())
        train(
            device_model,
            train_datas[i],
            val_datas[i],
            batch_size=128,
            lr=0.001,
            # checkpoint_save_path=f"./checkpoints/device_squeezenet_cifar10_0.25_finetuned_{round_num}_{i}.pth",
        )
        model_weights.append(device_model.state_dict())

    global_model.load_state_dict(
        fed_avg(model_weights, [len(train_data) for train_data in train_datas])
    )

    accuracy = test_accuracy(global_model, test_data)
    accuracy_list.append(accuracy)
    print(f"Round {round_num+1}: {accuracy}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(
            global_model.state_dict(),
            "./checkpoints/device_squeezenet_cifar10_0.2_aggregated.pth",
        )

# Print results
for i, accuracy in enumerate(accuracy_list):
    print(f"Round {i}: {accuracy}")
