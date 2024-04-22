import torch
import torchvision.transforms.v2 as T
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.models import squeezenet1_1

from modules.functional import fed_avg, test_accuracy, train
from modules.models import Ensemble

transform = T.Compose(
    [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

train_data = CIFAR10(root="./data", train=True, transform=transform)
test_data = CIFAR10(root="./data", train=False, transform=transform)


# Split the training data into several subsets
num_splits = 10
split_sizes = [len(train_data) // num_splits] * num_splits
split_sizes[-1] += len(train_data) % num_splits

train_data_splits = random_split(train_data, split_sizes)

val_data_ratio = 0.1
val_data_splits = []
for i, subset in enumerate(train_data_splits):
    val_size = int(val_data_ratio * len(subset))
    train_size = len(subset) - val_size
    train_subset, val_subset = random_split(subset, [train_size, val_size])
    train_data_splits[i] = train_subset
    val_data_splits.append(val_subset)


# Train models on each subset
model_global = squeezenet1_1(weights=None, num_classes=10)
models = []
model_weights = []
for i in range(num_splits):
    model = squeezenet1_1(weights=None, num_classes=10)
    model.load_state_dict(model_global.state_dict())
    train(
        model,
        train_data_splits[i],
        val_data_splits[i],
        lr=0.01,
        checkpoint_save_path=f"checkpoints/model_{i}.pth",
    )
    models.append(model)
    model_weights.append(model.state_dict())


# Compare model ensemble and FedAvg
ensemble = Ensemble(models)
accuracy_ensemble = test_accuracy(ensemble, test_data)

avg_weights = fed_avg(model_weights, split_sizes)
model_avg = squeezenet1_1(weights=None, num_classes=10)
model_avg.load_state_dict(avg_weights)
accuracy_avg = test_accuracy(model_avg, test_data)

print(f"Ensemble accuracy: {accuracy_ensemble}")
print(f"FedAvg accuracy: {accuracy_avg}")
