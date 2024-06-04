import torch

torch.set_float32_matmul_precision("high")

from lightning import Trainer
from modules.datasets import TinyImageNet200
from modules.functional import test_accuracy
from modules.lightning_modules import (
    LitModuleForFusion,
    LitModuleForImageClassification,
)
from modules.models import FusionModel, LowLevelEncoder, SimpleCNN
from torch.nn import Conv2d
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

transform = Compose(
    [ToTensor(), Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])]
)

dataset = TinyImageNet200(root="./data", split="train", transform=transform)

total_len = len(dataset)
cloud_len = int(total_len * 0.5)
party_len = (total_len - cloud_len) // 10
lengths = [cloud_len] + [party_len] * 9 + [total_len - cloud_len - party_len * 9]
subsets = random_split(dataset, lengths)
cloud_data, party_data_list = subsets[0], subsets[1:]

shared_encoder = LowLevelEncoder(in_channels=3, out_channels=16)

cloud_model = resnet18(weights=None, num_classes=200)
cloud_model.conv1 = Conv2d(
    16, 64, kernel_size=7, stride=(2, 2), padding=(3, 3), bias=False
)

party_models = [
    SimpleCNN(in_channels=16, input_size=64, num_classes=200) for _ in range(10)
]

control_model = SimpleCNN(in_channels=16, input_size=64, num_classes=200)

# Cloud pretrain
cloud_litmodule = LitModuleForFusion(
    cloud_model, control_model, shared_encoder, learning_rate=1e-4
)

trainer = Trainer(
    max_epochs=10,
    default_root_dir="checkpoints/cloud_pretrain",
    devices=1,
    accelerator="gpu",
    enable_checkpointing=False,
)
cloud_train_dataloader = DataLoader(
    cloud_data, batch_size=64, shuffle=True, num_workers=7
)
trainer.fit(model=cloud_litmodule, train_dataloaders=cloud_train_dataloader)

trained_shared_encoder = cloud_litmodule.shared_encoder
trained_cloud_model = cloud_litmodule.cloud_model
trained_control_model = cloud_litmodule.control_model

complete_model = FusionModel(
    trained_cloud_model, trained_control_model, trained_shared_encoder
)

test_dataset = TinyImageNet200(root="./data", split="val", transform=transform)
accuracy = test_accuracy(complete_model, test_dataset)
print(f"Test accuracy: {accuracy}")
