import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import Caltech101


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

        print(num_samples)

        # Randomly shuffle the indices
        np.random.shuffle(class_indices)

        # Assign the indices to each participant
        start = 0
        for j in range(num_participants):
            end = start + num_samples[j]
            participant_indices[j].extend(class_indices[start:end])
            start = end

    return [Subset(dataset, indices) for indices in participant_indices]
