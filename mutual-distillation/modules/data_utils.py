import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, random_split
from torchvision.datasets import CIFAR10, Caltech101, FashionMNIST, Food101


class ExtractedFeaturesDataset(Dataset):
    def __init__(self, original_dataset, feature_extractor, device="cuda"):
        super().__init__()
        self.original_dataset = original_dataset
        self.feature_extractor = feature_extractor
        self.device = device

    def __getitem__(self, index):
        data, target = self.original_dataset[index]
        data_original_device = data.device
        data = data.to(self.device)
        fe_original_device = next(self.feature_extractor.parameters()).device
        self.feature_extractor.to(self.device)
        data = data.unsqueeze(0)
        with torch.no_grad():
            data = self.feature_extractor(data)
        data = data.squeeze(0)
        data = data.to(data_original_device)
        self.feature_extractor.to(fe_original_device)
        return data, target

    def __len__(self):
        return len(self.original_dataset)


class MappedCaltech101(Caltech101):
    def __init__(self, offset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset = offset

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target + self.offset


def load_Caltech101(
    root="./data", download=True, transform=None, train_ratio=0.9, *args, **kwargs
):
    """Including:
    - transformations
    - train, validation, and test splits
    """
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    dataset = Caltech101(
        root=root, download=download, transform=transform, target_type="category"
    )

    original_dataset_length = len(dataset)

    train_data, test_data = random_split(
        dataset,
        [
            int(original_dataset_length * train_ratio),
            original_dataset_length - int(original_dataset_length * train_ratio),
        ],
    )

    return train_data, test_data


def load_Food101(
    root="./data", download=True, transform=None, val_ratio=0.1, *args, **kwargs
):
    """Perform default transformations and split the dataset into training, validation and test sets."""

    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    train_data = Food101(
        root=root,
        download=download,
        transform=transform,
        split="train",
        *args,
        **kwargs
    )

    original_train_data_length = len(train_data)
    train_data, val_data = random_split(
        train_data,
        [
            int(original_train_data_length * (1 - val_ratio)),
            original_train_data_length
            - int(original_train_data_length * (1 - val_ratio)),
        ],
    )

    test_data = Food101(
        root=root, download=download, transform=transform, split="test", *args, **kwargs
    )

    return train_data, val_data, test_data


def load_CIFAR10(
    root="./data", transform=None, train_ratio=0.9, download=False, *args, **kwargs
):
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    train_data = CIFAR10(
        root=root, train=True, transform=transform, download=download, *args, **kwargs
    )

    original_train_data_length = len(train_data)
    train_data_length = int(original_train_data_length * train_ratio)
    val_data_length = original_train_data_length - train_data_length
    train_data, val_data = random_split(
        train_data,
        [
            train_data_length,
            val_data_length,
        ],
    )

    test_data = CIFAR10(
        root=root, train=False, transform=transform, download=download, *args, **kwargs
    )

    return train_data, val_data, test_data


def load_FashionMNIST(
    root="./data", transform=None, train_ratio=0.9, download=False, *args, **kwargs
):
    if transform is None:
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

    train_data = FashionMNIST(
        root=root, train=True, transform=transform, download=download, *args, **kwargs
    )

    original_train_data_length = len(train_data)
    train_data_length = int(original_train_data_length * train_ratio)
    val_data_length = original_train_data_length - train_data_length
    train_data, val_data = random_split(
        train_data,
        [
            train_data_length,
            val_data_length,
        ],
    )

    test_data = FashionMNIST(
        root=root, train=False, transform=transform, download=download, *args, **kwargs
    )

    return train_data, val_data, test_data


def dirichlet_split(dataset, num_participants, num_classes=None):
    if num_classes is None:
        num_classes = len(dataset.classes)
    participant_indices = [[] for _ in range(num_participants)]
    class_proportions = np.random.dirichlet(
        np.ones(num_participants), size=num_classes
    )  # (num_classes, num_participants)

    labels = [y for _, y in dataset]

    for i in range(num_classes):
        # indices of samples whose label is i
        class_indices = np.where(np.array(labels) == i)[0]

        # Calculate the number of samples for each participant
        num_samples = [
            int(len(class_indices) * proportion) for proportion in class_proportions[i]
        ]

        # If there are any remaining samples, add them to the first participant
        num_samples[0] += len(class_indices) - sum(num_samples)

        # Randomly shuffle the indices
        np.random.shuffle(class_indices)

        # Assign the indices to each participant
        start = 0
        for j in range(num_participants):
            end = start + num_samples[j]
            participant_indices[j].extend(class_indices[start:end])
            start = end

    return [Subset(dataset, indices) for indices in participant_indices]
