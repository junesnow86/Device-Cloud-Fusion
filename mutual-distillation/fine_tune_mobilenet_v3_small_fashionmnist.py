import pickle

import torch
from torchvision.models import mobilenet_v3_small

from modules.evaluation import test_accuracy
from modules.models import ImageClassificationModel
from modules.training import train

with open("./checkpoints/data/participant_fashionmnist_train_data_0.pkl", "rb") as f:
    train_data = pickle.load(f)

with open("./checkpoints/data/participant_fashionmnist_val_data_0.pkl", "rb") as f:
    val_data = pickle.load(f)

with open("./checkpoints/data/participant_fashionmnist_test_data.pkl", "rb") as f:
    test_data = pickle.load(f)

backbone = mobilenet_v3_small(weights=None, num_classes=256)
model = ImageClassificationModel(backbone, 256, num_classes=10)
model.load_state_dict(torch.load("./checkpoints/custom_mobilenet_v3_small_cifar10.pth"))

before_accuracy = test_accuracy(model, test_data)
print(f"Test accuracy before fine-tuning: {before_accuracy:.4f}")

train(
    model,
    train_data,
    val_data,
    batch_size=128,
    lr=0.1,
    checkpoint_save_path="./checkpoints/custom_mobilenet_v3_small_fashionmnist_fine_tune.pth",
)

after_accuracy = test_accuracy(model, test_data)
print(f"Test accuracy after fine-tuning: {after_accuracy:.4f}")
