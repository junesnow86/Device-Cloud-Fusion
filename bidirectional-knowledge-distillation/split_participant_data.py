import pickle

import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.datasets import FashionMNIST

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为RGB图像
        transforms.Resize((32, 32)),  # 将图像大小调整为32x32
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


test_data = FashionMNIST(
    root="./data", train=False, transform=transform, download=False
)

with open("checkpoints/data/participant_fashionmnist_test_data.pkl", "wb") as f:
    pickle.dump(test_data, f)


all_train_data = FashionMNIST(
    root="./data", train=True, transform=transform, download=False
)

train_size = len(all_train_data)
train_sizes = [int(train_size / 3)] * 3
train_sizes[0] += train_size % 3

train_datasets = random_split(all_train_data, train_sizes)

train_datas, val_datas = [], []
for train_dataset in train_datasets:
    train_size = len(train_dataset)
    train_size = int(train_size * 0.9)
    val_size = len(train_dataset) - train_size
    train_data, val_data = random_split(train_dataset, [train_size, val_size])
    train_datas.append(train_data)
    val_datas.append(val_data)

with open("checkpoints/data/participant_fashionmnist_train_datas.pkl", "wb") as f:
    pickle.dump(train_datas, f)

with open("checkpoints/data/participant_fashionmnist_val_datas.pkl", "wb") as f:
    pickle.dump(val_datas, f)
