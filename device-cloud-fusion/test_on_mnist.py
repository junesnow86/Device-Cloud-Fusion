import lightning as L
import numpy as np
import torch

torch.set_float32_matmul_precision("high")

from modules.functional import test_accuracy
from modules.lightning_modules import LitModuleForImageClassification
from modules.models import SimpleCNN
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

dataset = MNIST(root="data", train=True, download=True, transform=ToTensor())
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=7)

model = SimpleCNN(in_channels=1, input_size=28, num_classes=10)
litmodule = LitModuleForImageClassification(model)

trainer = L.Trainer(
    max_epochs=10, default_root_dir="checkpoints/", devices=1, accelerator="gpu"
)
trainer.fit(model=litmodule, train_dataloaders=train_loader)

model = litmodule.model
model.eval()

test_dataset = MNIST(root="data", train=False, download=True, transform=ToTensor())
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
