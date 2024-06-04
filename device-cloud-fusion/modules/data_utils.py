import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, random_split
from torchvision.datasets import CIFAR10, Caltech101, FashionMNIST, Food101


class SubsetBasedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        data, target = self.subset[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.subset)


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


class RawData:
    def __init__(self, root, dataset="tiny-imagenet-200/") -> None:
        self.data_path = os.path.join(root, dataset)

        self.__labels_t_path = "%s%s" % (self.data_path, "wnids.txt")
        self.__train_data_path = "%s%s" % (self.data_path, "train/")
        self.__val_data_path = "%s%s" % (self.data_path, "val/")

        self.__labels_t = None
        self.__image_names = None

        self.__val_labels_t = None
        self.__val_labels = None
        self.__val_names = None

    @property
    def labels_t(self):
        if self.__labels_t is None:
            labels_t = []
            with open(self.__labels_t_path) as wnid:
                for line in wnid:
                    labels_t.append(line.strip("\n"))

            self.__labels_t = labels_t

        return self.__labels_t

    @property
    def image_names(self):
        if self.__image_names is None:
            image_names = []
            labels_t = self.labels_t
            for label in labels_t:
                txt_path = self.__train_data_path + label + "/" + label + "_boxes.txt"
                image_name = []
                with open(txt_path) as txt:
                    for line in txt:
                        image_name.append(line.strip("\n").split("\t")[0])
                image_names.append(image_name)

            self.__image_names = image_names

        return self.__image_names

    @property
    def val_labels_t(self):
        if self.__val_labels_t is None:
            val_labels_t = []
            with open(self.__val_data_path + "val_annotations.txt") as txt:
                for line in txt:
                    val_labels_t.append(line.strip("\n").split("\t")[1])

            self.__val_labels_t = val_labels_t

        return self.__val_labels_t

    @property
    def val_names(self):
        if self.__val_names is None:
            val_names = []
            with open(self.__val_data_path + "val_annotations.txt") as txt:
                for line in txt:
                    val_names.append(line.strip("\n").split("\t")[0])

            self.__val_names = val_names

        return self.__val_names

    @property
    def val_labels(self):
        if self.__val_labels is None:
            val_labels = []
            val_labels_t = self.val_labels_t
            labels_t = self.labels_t
            for i in range(len(val_labels_t)):
                for i_t in range(len(labels_t)):
                    if val_labels_t[i] == labels_t[i_t]:
                        val_labels.append(i_t)
            val_labels = np.array(val_labels)

            self.__val_labels = val_labels

        return self.__val_labels


class TinyImageNet200(Dataset):

    def __init__(self, root, split, transform):
        """
        split: `train` or `val`
        """
        data_path = os.path.join(root, "tiny-imagenet-200/")
        self.__train_data_path = "%s%s" % (data_path, "train/")
        self.__val_data_path = "%s%s" % (data_path, "val/")

        self.split = split

        self.raw_data = RawData(root)

        self.labels_t = self.raw_data.labels_t
        self.image_names = self.raw_data.image_names
        self.val_names = self.raw_data.val_names

        self.transform = transform

    def __getitem__(self, index):
        label = None
        image = None

        labels_t = self.labels_t
        image_names = self.image_names
        val_labels = self.raw_data.val_labels
        val_names = self.val_names

        if self.split == "train":
            label = index // 500  # 500 images per class
            remain = index % 500
            image_path = os.path.join(
                self.__train_data_path,
                labels_t[label],
                "images",
                image_names[label][remain],
            )
            image = cv2.imread(image_path)
            image = np.array(image).reshape(64, 64, 3)

        elif self.split == "val":
            label = val_labels[index]
            val_image_path = os.path.join(
                self.__val_data_path, "images", val_names[index]
            )
            image = np.array(cv2.imread(val_image_path)).reshape(64, 64, 3)

        return self.transform(image), label

    def __len__(self):
        len_ = 0
        if self.split == "train":
            len_ = len(self.image_names) * len(self.image_names[0])
        elif self.split == "val":
            len_ = len(self.val_names)

        return len_


if __name__ == "__main__":
    dataset = TinyImageNet200(
        root="/home/ljt/Device-Cloud-Fusion/bidirectional-knowledge-distillation/data",
        split="train",
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    print(len(dataset))
