import torch
from torchvision.models import (
    mobilenet_v3_small,
    resnet18,
    shufflenet_v2_x1_0,
    squeezenet1_0,
)

from modules.data_utils import load_CIFAR10
from modules.distillation import distill
from modules.evaluation import test_accuracy

cloud_model = resnet18(weights=None, num_classes=10)
cloud_model.load_state_dict(torch.load("./checkpoints/resnet18_cifar10.pth"))

participant_model0 = mobilenet_v3_small(weights=None, num_classes=10)
participant_model0.load_state_dict(
    torch.load("./checkpoints/mobilenet_v3_small_fashionmnist_fine_tune.pth")
)
participant_model1 = shufflenet_v2_x1_0(weights=None, num_classes=10)
participant_model1.load_state_dict(
    torch.load("./checkpoints/shufflenet_v2_x1_0_fashionmnist_fine_tune.pth")
)
participant_model2 = squeezenet1_0(weights=None, num_classes=10)
participant_model2.load_state_dict(
    torch.load("./checkpoints/squeezenet1_0_fashionmnist_fine_tune.pth")
)

transfer_train_data, transfer_val_data, transfer_test_data = load_CIFAR10(
    root="./data", download=False
)

cloud_accuracy_before_distillation = test_accuracy(cloud_model, transfer_test_data)
print(f"Test accuracy of ResNet18: {cloud_accuracy_before_distillation:.4f}")
