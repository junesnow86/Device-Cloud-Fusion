import pickle

import torch
from torchvision.models import shufflenet_v2_x1_0

from modules.evaluation import test_accuracy
from modules.training import train

with open("./checkpoints/data/participant_fashionmnist_train_data_1.pkl", "rb") as f:
    train_data = pickle.load(f)

with open("./checkpoints/data/participant_fashionmnist_val_data_1.pkl", "rb") as f:
    val_data = pickle.load(f)

with open("./checkpoints/data/participant_fashionmnist_test_data.pkl", "rb") as f:
    test_data = pickle.load(f)

model = shufflenet_v2_x1_0(weights=None, num_classes=10)
model.load_state_dict(torch.load("./checkpoints/shufflenet_v2_x1_0_cifar10.pth"))

before_accuracy = test_accuracy(model, test_data)
print(f"Test accuracy before fine-tuning: {before_accuracy:.4f}")

train(
    model,
    train_data,
    val_data,
    batch_size=128,
    lr=0.1,
    checkpoint_save_path="./checkpoints/shufflenet_v2_x1_0_fashionmnist_fine_tune.pth",
)

after_accuracy = test_accuracy(model, test_data)
print(f"Test accuracy after fine-tuning: {after_accuracy:.4f}")
