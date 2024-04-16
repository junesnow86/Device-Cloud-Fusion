import torch
from torchvision.models import resnet18, shufflenet_v2_x1_0

from modules.data_utils import load_CIFAR10
from modules.distillation import distill
from modules.evaluation import test_accuracy
from modules.models import ImageClassificationModel

cloud_backbone = resnet18(weights=None, num_classes=256)
cloud_model = ImageClassificationModel(cloud_backbone, 256, num_classes=10)
cloud_model.load_state_dict(torch.load("./checkpoints/custom_resnet18_cifar10.pth"))

participant_backbone = shufflenet_v2_x1_0(weights=None, num_classes=256)
participant_model = ImageClassificationModel(participant_backbone, 256, num_classes=10)

transfer_train_data, transfer_val_data, transfer_test_data = load_CIFAR10(
    root="./data", download=False
)

cloud_accuracy = test_accuracy(cloud_model, transfer_test_data)
print(f"Test accuracy of ResNet18: {cloud_accuracy:.4f}")

distill(
    participant_model,
    cloud_model,
    transfer_train_data,
    transfer_val_data,
    batch_size=64,
    lr=0.01,
    checkpoint_save_path="./checkpoints/custom_shufflenet_v2_x1_0_cifar10.pth",
)
accuracy = test_accuracy(participant_model, transfer_test_data)


print(f"Test accuracy of ResNet18: {cloud_accuracy:.4f}")
print(f"Test accuracy of MobileNetV3-Small: {accuracy:.4f}")
