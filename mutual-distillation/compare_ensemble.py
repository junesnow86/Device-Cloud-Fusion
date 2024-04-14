import pickle

import torch
from torchvision.models import mobilenet_v3_small, shufflenet_v2_x1_0, squeezenet1_0

from modules.data_utils import load_CIFAR10
from modules.ensemble import Ensemble
from modules.evaluation import test_accuracy, test_accuracy_with_preds
from modules.models import ImageClassificationModel

# model0 = ImageClassificationModel(
#     mobilenet_v3_small(weights=None, num_classes=256), 256, num_classes=10
# )
# model0.load_state_dict(
#     torch.load("./checkpoints/custom_mobilenet_v3_small_cifar10.pth")
# )

# model1 = ImageClassificationModel(
#     shufflenet_v2_x1_0(weights=None, num_classes=256), 256, num_classes=10
# )
# model1.load_state_dict(
#     torch.load("./checkpoints/custom_shufflenet_v2_x1_0_cifar10.pth")
# )

# model2 = ImageClassificationModel(
#     squeezenet1_0(weights=None, num_classes=256), 256, num_classes=10
# )
# model2.load_state_dict(torch.load("./checkpoints/custom_squeezenet1_0_cifar10.pth"))

# ensemble = Ensemble([model0, model1, model2])
# ensemble_vote = Ensemble([model0, model1, model2], mode="voting")

# train_data, val_data, test_data = load_CIFAR10(root="./data", download=False)

# accuracy0 = test_accuracy(model0, test_data)
# accuracy1 = test_accuracy(model1, test_data)
# accuracy2 = test_accuracy(model2, test_data)
# ensemble_accuracy = test_accuracy(ensemble, test_data)
# ensemble_vote_accuracy = test_accuracy_with_preds(ensemble_vote, test_data)

# print(f"Model 0 accuracy: {accuracy0:.4f}")
# print(f"Model 1 accuracy: {accuracy1:.4f}")
# print(f"Model 2 accuracy: {accuracy2:.4f}")
# print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
# print(f"Ensemble vote accuracy: {ensemble_vote_accuracy:.4f}")

with open(
    "checkpoints/data/participant_fashionmnist_test_data.pkl",
    "rb",
) as f:
    test_data = pickle.load(f)

model0_fine_tune = ImageClassificationModel(
    mobilenet_v3_small(weights=None, num_classes=256), 256, num_classes=10
)
model0_fine_tune.load_state_dict(
    torch.load("./checkpoints/custom_mobilenet_v3_small_fashionmnist_fine_tune.pth")
)

model1_fine_tune = ImageClassificationModel(
    shufflenet_v2_x1_0(weights=None, num_classes=256), 256, num_classes=10
)
model1_fine_tune.load_state_dict(
    torch.load("./checkpoints/custom_shufflenet_v2_x1_0_fashionmnist_fine_tune.pth")
)

model2_fine_tune = ImageClassificationModel(
    squeezenet1_0(weights=None, num_classes=256), 256, num_classes=10
)
model2_fine_tune.load_state_dict(
    torch.load("./checkpoints/custom_squeezenet1_0_fashionmnist_fine_tune.pth")
)

ensemble_fine_tune = Ensemble([model0_fine_tune, model1_fine_tune, model2_fine_tune])
ensemble_vote_fine_tune = Ensemble(
    [model0_fine_tune, model1_fine_tune, model2_fine_tune], mode="voting"
)

accuracy0_fine_tune = test_accuracy(model0_fine_tune, test_data)
accuracy1_fine_tune = test_accuracy(model1_fine_tune, test_data)
accuracy2_fine_tune = test_accuracy(model2_fine_tune, test_data)
ensemble_fine_tune_accuracy = test_accuracy(ensemble_fine_tune, test_data)
ensemble_vote_fine_tune_accuracy = test_accuracy_with_preds(
    ensemble_vote_fine_tune, test_data
)

print(f"Model 0 fine-tune accuracy: {accuracy0_fine_tune:.4f}")
print(f"Model 1 fine-tune accuracy: {accuracy1_fine_tune:.4f}")
print(f"Model 2 fine-tune accuracy: {accuracy2_fine_tune:.4f}")
print(f"Ensemble fine-tune accuracy: {ensemble_fine_tune_accuracy:.4f}")
print(f"Ensemble vote fine-tune accuracy: {ensemble_vote_fine_tune_accuracy:.4f}")
