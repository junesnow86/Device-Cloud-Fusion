import pickle

import torch
from torch.utils.data import Subset, random_split
from torchvision.datasets import CIFAR10

dataset = CIFAR10(root="./data", train=True, download=False)
targets = torch.tensor(dataset.targets)
classes, class_counts = targets.unique(return_counts=True)
cloud_ratio = 0.4
cloud_sample_per_class = (class_counts * cloud_ratio).int()
cloud_sample_indices = []

for i, c in enumerate(classes):
    class_indices = (targets == c).nonzero().view(-1)
    class_sample_indices = class_indices[
        torch.randperm(len(class_indices))[: cloud_sample_per_class[i]]
    ]
    cloud_sample_indices.extend(class_sample_indices.tolist())

cloud_subset = Subset(dataset, cloud_sample_indices)

remaining_indices = list(set(range(len(dataset))) - set(cloud_sample_indices))
remaining_subset = Subset(dataset, remaining_indices)

with open("data/cloud_subset_cifar10_0.4.pkl", "wb") as f:
    pickle.dump(cloud_subset, f)

with open("data/remaining_subset_cifar10_0.6.pkl", "wb") as f:
    pickle.dump(remaining_subset, f)

# # Calculate the number of samples in each class for the cloud_subset
# cloud_targets = torch.tensor([dataset.targets[i] for i in cloud_sample_indices])
# cloud_classes, cloud_class_counts = cloud_targets.unique(return_counts=True)
# print(
#     "Cloud subset class counts:",
#     dict(zip(cloud_classes.tolist(), cloud_class_counts.tolist())),
# )

# # Calculate the number of samples in each class for the remaining_subset
# remaining_targets = torch.tensor([dataset.targets[i] for i in remaining_indices])
# remaining_classes, remaining_class_counts = remaining_targets.unique(return_counts=True)
# print(
#     "Remaining subset class counts:",
#     dict(zip(remaining_classes.tolist(), remaining_class_counts.tolist())),
# )


size0 = len(remaining_subset) // 3
size1 = size0
size2 = len(remaining_subset) - size0 - size1

subset0, subset1, subset2 = random_split(remaining_subset, [size0, size1, size2])

print(len(remaining_subset))
print(len(subset0))
print(len(subset1))
print(len(subset2))

with open("data/remaining_subset0_cifar10_0.2.pkl", "wb") as f:
    pickle.dump(subset0, f)

with open("data/remaining_subset1_cifar10_0.2.pkl", "wb") as f:
    pickle.dump(subset1, f)

with open("data/remaining_subset2_cifar10_0.2.pkl", "wb") as f:
    pickle.dump(subset2, f)

# Calculate the number of samples in each class for the remaining_subset
remaining_targets = torch.tensor(
    [remaining_subset.dataset.targets[i] for i in remaining_subset.indices]
)
remaining_classes, remaining_class_counts = remaining_targets.unique(return_counts=True)
print(
    "Remaining subset class counts:",
    dict(zip(remaining_classes.tolist(), remaining_class_counts.tolist())),
)

remaining_targets_0 = torch.tensor(
    [remaining_subset.dataset.targets[i] for i in subset0.indices]
)
remaining_classes_0, remaining_class_counts_0 = remaining_targets_0.unique(
    return_counts=True
)
print(
    "Remaining subset0 class counts:",
    dict(zip(remaining_classes_0.tolist(), remaining_class_counts_0.tolist())),
)

remaining_targets_1 = torch.tensor(
    [remaining_subset.dataset.targets[i] for i in subset1.indices]
)
remaining_classes_1, remaining_class_counts_1 = remaining_targets_1.unique(
    return_counts=True
)
print(
    "Remaining subset1 class counts:",
    dict(zip(remaining_classes_1.tolist(), remaining_class_counts_1.tolist())),
)

remaining_targets_2 = torch.tensor(
    [remaining_subset.dataset.targets[i] for i in subset2.indices]
)
remaining_classes_2, remaining_class_counts_2 = remaining_targets_2.unique(
    return_counts=True
)
print(
    "Remaining subset2 class counts:",
    dict(zip(remaining_classes_2.tolist(), remaining_class_counts_2.tolist())),
)

print(set(subset0.indices) & set(subset1.indices))
print(set(subset0.indices) & set(subset2.indices))
print(set(subset1.indices) & set(subset2.indices))
