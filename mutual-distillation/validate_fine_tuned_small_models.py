import torch
from torchvision.models import mobilenet_v3_small, shufflenet_v2_x1_0, squeezenet1_0

from modules.data_utils import load_CIFAR10
from modules.ensemble import Ensemble
from modules.evaluation import test_accuracy, test_accuracy_with_preds
from modules.models import ImageClassificationModel

transfer_train_data, transfer_val_data, transfer_test_data = load_CIFAR10(
    root="./data", download=False
)

participant_backbone0 = mobilenet_v3_small(weights=None, num_classes=256)
participant_model0_fine_tune = ImageClassificationModel(
    participant_backbone0, 256, num_classes=10
)
participant_model0_fine_tune.load_state_dict(
    torch.load("./checkpoints/custom_mobilenet_v3_small_fashionmnist_fine_tune.pth")
)

participant_backbone0 = mobilenet_v3_small(weights=None, num_classes=256)
participant_model0 = ImageClassificationModel(
    participant_backbone0, 256, num_classes=10
)
participant_model0.load_state_dict(
    torch.load("./checkpoints/custom_mobilenet_v3_small_cifar10.pth")
)
pretrained_accuracy = test_accuracy(participant_model0, transfer_test_data)

participant_model0.backbone = participant_model0_fine_tune.backbone
participant_model0_accuracy = test_accuracy(participant_model0, transfer_test_data)

participant_backbone1 = shufflenet_v2_x1_0(weights=None, num_classes=256)
participant_model1_fine_tune = ImageClassificationModel(
    participant_backbone1, 256, num_classes=10
)
participant_model1_fine_tune.load_state_dict(
    torch.load("./checkpoints/custom_shufflenet_v2_x1_0_fashionmnist_fine_tune.pth")
)
participant_backbone1 = shufflenet_v2_x1_0(weights=None, num_classes=256)
participant_model1 = ImageClassificationModel(
    participant_backbone1, 256, num_classes=10
)
participant_model1.load_state_dict(
    torch.load("./checkpoints/custom_shufflenet_v2_x1_0_cifar10.pth")
)
participant_model1.backbone = participant_model1_fine_tune.backbone
participant_model1_accuracy = test_accuracy(participant_model1, transfer_test_data)


participant_backbone2 = squeezenet1_0(weights=None, num_classes=256)
participant_model2_fine_tune = ImageClassificationModel(
    participant_backbone2, 256, num_classes=10
)
participant_model2_fine_tune.load_state_dict(
    torch.load("./checkpoints/custom_squeezenet1_0_fashionmnist_fine_tune.pth")
)
participant_backbone2 = squeezenet1_0(weights=None, num_classes=256)
participant_model2 = ImageClassificationModel(
    participant_backbone2, 256, num_classes=10
)
participant_model2.load_state_dict(
    torch.load("./checkpoints/custom_squeezenet1_0_cifar10.pth")
)
participant_model2.backbone = participant_model2_fine_tune.backbone
participant_model2_accuracy = test_accuracy(participant_model2, transfer_test_data)


ensemble = Ensemble([participant_model0, participant_model1, participant_model2])
ensemble_accuracy = test_accuracy(ensemble, transfer_test_data)

print(f"Test accuracy of participant model 0: {participant_model0_accuracy:.4f}")
print(f"Test accuracy of participant model 1: {participant_model1_accuracy:.4f}")
print(f"Test accuracy of participant model 2: {participant_model2_accuracy:.4f}")
print(f"Test accuracy of the ensemble: {ensemble_accuracy:.4f}")
