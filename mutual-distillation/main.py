import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from data import ExtractedFeaturesDataset
from models import Classifier, FeatureExtractor
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small, resnet152
from tqdm import tqdm


def train(model, train_data, val_data, epochs=50, device="cuda", patience=3):
    model.to(device)
    model.train()
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_loss = float("inf")
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_dataloader, desc="training"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        with torch.no_grad():
            val_loss = 0.0
            for inputs, labels in tqdm(val_dataloader, desc="validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(
            f"Epoch {epoch+1}: training loss={running_loss/len(train_dataloader)}, validation loss={val_loss/len(val_dataloader)}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 3
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stop at epoch {epoch+1}.")
                break


def test(model, data, device="cuda"):
    model.to(device)
    model.eval()
    dataloader = DataLoader(data, batch_size=64, shuffle=False)
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def distill(
    teacher, student, train_data, finetune_data, epochs=5, device="cuda", patience=3
):
    """
    train_data: the data used to teach the student model
    finetune_data: the date used to fine-tune the student model
    """

    def loss_fn_kd(
        outputs, teacher_outputs, labels=None, T=1, alpha=0.5, with_labels=False
    ):
        soft_loss = nn.KLDivLoss(reduction="batchmean")(
            nn.functional.log_softmax(outputs / T, dim=1),
            nn.functional.softmax(teacher_outputs / T, dim=1),
        ) * (T * T)
        if with_labels:
            hard_loss = nn.CrossEntropyLoss()(outputs, labels) * (1.0 - alpha)
            soft_loss = soft_loss * alpha + hard_loss
        return soft_loss

    teacher.to(device)
    student.to(device)
    teacher.eval()
    student.train()
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    finetune_dataloader = DataLoader(finetune_data, batch_size=64, shuffle=True)
    finetune_loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(student.parameters(), lr=0.001, momentum=0.9)

    # best_loss = float('inf')
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_dataloader, desc="distillation training"):
            # use teacher logits as soft targets
            inputs = inputs.to(device)
            labels = labels.to(device)
            teacher_outputs = teacher(inputs)
            student_outputs = student(inputs)
            loss = loss_fn_kd(student_outputs, teacher_outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        for inputs, labels in tqdm(
            finetune_dataloader, desc="distillation fine-tuning"
        ):
            # use hard labels to avoid catastrophic forgetting
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = student(inputs)
            loss = finetune_loss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # val_loss = 0.0
        # with torch.no_grad():
        #     for inputs, labels in tqdm(val_dataloader, desc="distillation validation"):
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)
        #         teacher_outputs = teacher(inputs)
        #         student_outputs = student(inputs)
        #         loss = loss_fn(student_outputs, teacher_outputs)
        #         val_loss += loss.item()

        print(f"Epoch {epoch+1}: training loss={running_loss/len(train_dataloader)}")
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     patience = 3
        # else:
        #     patience -= 1
        #     if patience == 0:
        #         print(f"Early stop at epoch {epoch+1}.")
        #         break


# Load CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
train_data = CIFAR10(root="./data", train=True, download=True, transform=transform)
test_data = CIFAR10(root="./data", train=False, download=True, transform=transform)

# Split data into cloud part and parties part
cloud_size = len(train_data) // 10
participant_size = len(train_data) - cloud_size
cloud_train, parties_train = random_split(train_data, [cloud_size, participant_size])

# Split participant data according to the class
labels = [y for _, y in parties_train]
indices = np.array(range(len(labels)))

partyA_classes = [0, 1, 2, 3, 4]
partyB_classes = [5, 6, 7, 8, 9]
# partyA_classes = list(range(50))
# partyB_classes = list(range(50, 100))

partyA_indices = np.where(np.isin(labels, partyA_classes))[0]
partyB_indices = np.where(np.isin(labels, partyB_classes))[0]

partyA_train = Subset(parties_train, partyA_indices)
partyB_train = Subset(parties_train, partyB_indices)

# Split validation data
cloud_train_size = len(cloud_train) * 8 // 10
cloud_val_size = len(cloud_train) - cloud_train_size
cloud_train, cloud_val = random_split(cloud_train, [cloud_train_size, cloud_val_size])

partyA_train_size = len(partyA_train) * 8 // 10
partyA_val_size = len(partyA_train) - partyA_train_size
partyA_train, partyA_val = random_split(
    partyA_train, [partyA_train_size, partyA_val_size]
)

partyB_train_size = len(partyB_train) * 8 // 10
partyB_val_size = len(partyB_train) - partyB_train_size
partyB_train, partyB_val = random_split(
    partyB_train, [partyB_train_size, partyB_val_size]
)


# Define models
cloud_fe = FeatureExtractor()
cloud_clf = Classifier(256, 100)
cloud_backbone = resnet152(weights=None, num_classes=256)
cloud_model = nn.Sequential(cloud_fe, cloud_backbone, cloud_clf)

party_A_fe = FeatureExtractor()
party_A_clf = Classifier(256, 100)
party_A_backbone = mobilenet_v3_small(weights=None, num_classes=256)
party_A_model = nn.Sequential(party_A_fe, party_A_backbone, party_A_clf)

party_B_fe = FeatureExtractor()
party_B_clf = Classifier(256, 100)
party_B_backbone = mobilenet_v3_large(weights=None, num_classes=256)
party_B_model = nn.Sequential(party_B_fe, party_B_backbone, party_B_clf)

cloud_model = resnet152(weights=None, num_classes=10)
party_A_model = mobilenet_v3_small(weights=None, num_classes=10)
party_B_model = mobilenet_v3_large(weights=None, num_classes=10)

# Train models respectively with the splitted dataset
print(">>> Start training cloud model...")
train(cloud_model, cloud_train, cloud_val)
cloud_acc = test(cloud_model, test_data)

print(">>> Start training party A model...")
train(party_A_model, partyA_train, partyA_val)
party_A_acc = test(party_A_model, test_data)

print(">>> Start training party B model...")
train(party_B_model, partyB_train, partyB_val)
party_B_acc = test(party_B_model, test_data)

# Perform knowledge distillation from parties to the cloud
print(">>> Start distillation from party A to cloud...")
# distill(nn.Sequential(feature_extractor, party_A_backbone), nn.Sequential(feature_extractor, cloud_backbone), party_A_train, party_A_val, loss_fn_kd)
# distil_train_data_A = ExtractedFeaturesDataset(partyA_train, party_A_fe)
# distil_val_data_A = ExtractedFeaturesDataset(partyA_val, party_A_fe)
# distil_cloud_data = ExtractedFeaturesDataset(cloud_train, cloud_fe)
# distill(party_A_backbone, cloud_backbone, distil_train_data_A, distil_cloud_data)
distill(party_A_model, cloud_model, partyA_train, cloud_train)
cloud_acc_kd1 = test(cloud_model, test_data)

# print(">>> Start fine-tuning cloud model...")
# train(cloud_model, cloud_data, epochs=10)  # fine-tuning
# cloud_acc_finetune1 = test(cloud_model, test_dataset)

print(">>> Start distillation from party B to cloud...")
# distill(nn.Sequential(feature_extractor, party_B_backbone), nn.Sequential(feature_extractor, cloud_backbone), party_B_train, party_B_val, loss_fn_kd)
# distil_train_data_B = ExtractedFeaturesDataset(partyB_train, party_B_fe)
# distil_val_data_B = ExtractedFeaturesDataset(partyB_val, party_B_fe)
# distill(party_B_model, cloud_model, distil_train_data_B, distil_cloud_data)
distill(party_B_model, cloud_model, partyB_train, cloud_train)
cloud_acc_kd2 = test(cloud_model, test_data)

# print(">>> Start fine-tuning cloud model...")
# train(cloud_model, cloud_data, epochs=10)  # fine-tuning
# cloud_acc_finetune2 = test(cloud_model, test_dataset)

# Perform KD from cloud backbone to parties
print(">>> Start distillation from cloud to party A...")
# distill(cloud_backbone, party_A_backbone, distil_cloud_data, distil_train_data_A)
distill(cloud_model, party_A_model, cloud_train, partyA_train)
party_A_acc_kd = test(party_A_model, test_data)

print(">>> Start distillation from cloud to party B...")
# distill(cloud_backbone, party_B_backbone, distil_cloud_data, distil_train_data_B)
distill(cloud_model, party_B_model, cloud_train, partyB_train)
party_B_acc_kd = test(party_B_model, test_data)

# Print results
print(f"Cloud model accuracy: {cloud_acc:.4f}")
print(f"Party A model accuracy: {party_A_acc:.4f}")
print(f"Party B model accuracy: {party_B_acc:.4f}")
print(f"Cloud model accuracy after distillation from party A: {cloud_acc_kd1:.4f}")
print(f"Cloud model accuracy after distillation from party B: {cloud_acc_kd2:.4f}")
print(f"Party A accuracy after distillation from cloud: {party_A_acc_kd:.4f}")
print(f"Party B accuracy after distillation from cloud: {party_B_acc_kd:.4f}")

# TODO:
# - 共享特征提取器和分类器层，但是需要处理non-iid所导致的参数加权平均效果不理想的问题
# - 修改数据分布，参与方不是垄断某些类，而是每个类都有一定比例的数据
