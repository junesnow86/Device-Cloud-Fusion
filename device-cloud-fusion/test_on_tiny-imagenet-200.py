import lightning as L
import numpy as np
import torch

torch.set_float32_matmul_precision("high")

from modules.datasets import TinyImageNet200
from modules.functional import test_accuracy
from modules.lightning_modules import LitModuleForImageClassification
from modules.models import SimpleCNN
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.transforms import Compose, Normalize, ToTensor

transform = Compose(
    [ToTensor(), Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])]
)

dataset = TinyImageNet200(root="./data", split="train", transform=transform)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=7)

model = SimpleCNN(in_channels=3, input_size=64, num_classes=200)
# model = resnet18(weights=None, num_classes=200)
# model = resnet34(weights=None, num_classes=200)
# model = resnet50(weights=None, num_classes=200)
litmodule = LitModuleForImageClassification(model, learning_rate=1e-4)

trainer = L.Trainer(
    max_epochs=10,
    default_root_dir="checkpoints/lightning_logs/resnet18_tiny-imagenet-200",
    devices=1,
    accelerator="gpu",
    enable_checkpointing=False,
)
trainer.fit(model=litmodule, train_dataloaders=train_loader)

model = litmodule.model
model.eval()

test_dataset = TinyImageNet200(root="./data", split="val", transform=transform)
accuracy = test_accuracy(model, test_dataset)
print(f"Test accuracy: {accuracy}")

indices = np.random.choice(len(test_dataset), 10, replace=False)
subset = Subset(test_dataset, indices)

subset_loader = DataLoader(subset, batch_size=10)

inputs, labels = next(iter(subset_loader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = inputs.to(device)
labels = labels.to(device)

outputs = model(inputs)
_, predicted = torch.max(outputs, 1)

print("Predicted:", predicted)
print("Ground truth:", labels)
