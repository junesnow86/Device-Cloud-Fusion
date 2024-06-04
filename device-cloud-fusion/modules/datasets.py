import os
import numpy as np
import cv2

from torch.utils.data import Dataset


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
