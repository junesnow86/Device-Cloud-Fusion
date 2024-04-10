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

participant_model1 = mobilenet_v3_small(weights=None, num_classes=10)
participant_model2 = squeezenet1_0(weights=None, num_classes=10)
participant_model3 = shufflenet_v2_x1_0(weights=None, num_classes=10)

transfer_train_data, transfer_val_data, transfer_test_data = load_CIFAR10(
    root="./data", download=False
)

cloud_accuracy = test_accuracy(cloud_model, transfer_test_data)
print(f"Test accuracy of ResNet18: {cloud_accuracy:.4f}")

distill(
    participant_model1,
    cloud_model,
    transfer_train_data,
    transfer_val_data,
    batch_size=64,
    lr=0.01,
    checkpoint_save_path="./checkpoints/mobilenet_v3_small_cifar10.pth",
)
accuracy1 = test_accuracy(participant_model1, transfer_test_data)

distill(
    participant_model2,
    cloud_model,
    transfer_train_data,
    transfer_val_data,
    batch_size=64,
    lr=0.01,
    checkpoint_save_path="./checkpoints/squeezenet1_0_cifar10.pth",
)
accuracy2 = test_accuracy(participant_model2, transfer_test_data)

distill(
    participant_model3,
    cloud_model,
    transfer_train_data,
    transfer_val_data,
    batch_size=64,
    lr=0.01,
    checkpoint_save_path="./checkpoints/shufflenet_v2_x1_0_cifar10.pth",
)
accuracy3 = test_accuracy(participant_model3, transfer_test_data)

print(f"Test accuracy of ResNet18: {cloud_accuracy:.4f}")
print(f"Test accuracy of MobileNetV3-Small: {accuracy1:.4f}")
print(f"Test accuracy of SqueezeNet1.0: {accuracy2:.4f}")
print(f"Test accuracy of ShuffleNetV2_x1_0: {accuracy3:.4f}")
