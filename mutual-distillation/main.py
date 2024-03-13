import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR10
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, resnet152
from tqdm import tqdm

from models import FeatureExtractor, Classifier


def loss_fn_kd(outputs, teacher_outputs, labels=None, T=20, alpha=0.5, with_labels=False):
    soft_loss = nn.KLDivLoss(reduction="batchmean")(nn.functional.log_softmax(outputs/T, dim=1),
                               nn.functional.softmax(teacher_outputs/T, dim=1)) * (T * T)
    if with_labels:
        hard_loss = nn.CrossEntropyLoss()(outputs, labels) * (1. - alpha)
        soft_loss = soft_loss * alpha + hard_loss
    return soft_loss

def train(model, train_data, val_data, epochs=50, device='cuda', patience=3):
    model.to(device)
    model.train()
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_loss = float('inf')
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
        
        print(f'Epoch {epoch+1}: training loss={running_loss/len(train_dataloader)}, validation loss={val_loss/len(val_dataloader)}')
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 3
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stop at epoch {epoch+1}.")
                break

def test(model, data, device='cuda'):
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

def distill(teacher, student, train_data, val_data, loss_fn, epochs=10, device='cuda', patience=3):
    teacher.to(device)
    student.to(device)
    teacher.eval()
    student.train()
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False)
    optimizer = optim.SGD(student.parameters(), lr=0.001, momentum=0.9)

    best_loss = float('inf')
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_dataloader, desc="distillation training"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            teacher_outputs = teacher(inputs)
            student_outputs = student(inputs)
            loss = loss_fn(student_outputs, teacher_outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_dataloader, desc="distillation validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                teacher_outputs = teacher(inputs)
                student_outputs = student(inputs)
                loss = loss_fn(student_outputs, teacher_outputs)
                val_loss += loss.item()

        print(f'Epoch {epoch+1}: training loss={running_loss/len(train_dataloader)}, validation loss={val_loss/len(val_dataloader)}')
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 3
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stop at epoch {epoch+1}.")
                break

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split dataset according to the class
class_indices = [np.where(np.array(train_data.targets) == i)[0] for i in range(10)]
cloud_indices = np.concatenate(class_indices[:2])
party_A_indices = np.concatenate(class_indices[2:6])
party_B_indices = np.concatenate(class_indices[6:])
cloud_data = Subset(train_data, cloud_indices)
party_A_data = Subset(train_data, party_A_indices)
party_B_data = Subset(train_data, party_B_indices)

num_train = len(cloud_data)
num_val = int(num_train * 0.1)
num_train = num_train - num_val
cloud_train, cloud_val = random_split(cloud_data, [num_train, num_val])
num_train = len(party_A_data)
num_val = int(num_train * 0.1)
num_train = num_train - num_val
party_A_train, party_A_val = random_split(party_A_data, [num_train, num_val])
num_train = len(party_B_data)
num_val = int(num_train * 0.1)
num_train = num_train - num_val
party_B_train, party_B_val = random_split(party_B_data, [num_train, num_val])

# Define models
feature_extractor = FeatureExtractor()
classifier = Classifier(256, 10)
cloud_backbone = resnet152(weights=None, num_classes=256)
cloud_model = nn.Sequential(feature_extractor, cloud_backbone, classifier)
party_A_backbone = mobilenet_v3_small(weights=None, num_classes=256)
party_A_model = nn.Sequential(feature_extractor, party_A_backbone, classifier)
party_B_backbone = mobilenet_v3_large(weights=None, num_classes=256)
party_B_model = nn.Sequential(feature_extractor, party_B_backbone, classifier)

# Train models respectively with the splitted dataset
print(">>> Start training cloud model...")
train(cloud_model, cloud_train, cloud_val)
cloud_acc = test(cloud_model, test_data)

print(">>> Start training party A model...")
train(party_A_model, party_A_train, party_A_val)
party_A_acc = test(party_A_model, test_data)

print(">>> Start training party B model...")
train(party_B_model, party_B_train, party_B_val)
party_B_acc = test(party_B_model, test_data)

# Perform knowledge distillation from parties to the cloud
print(">>> Start distillation from party A to cloud...")
# distill(nn.Sequential(feature_extractor, party_A_backbone), nn.Sequential(feature_extractor, cloud_backbone), party_A_train, party_A_val, loss_fn_kd)
distill(party_A_model, cloud_model, party_A_train, party_A_val, loss_fn_kd)
cloud_acc_kd1 = test(cloud_model, test_data)

# print(">>> Start fine-tuning cloud model...")
# train(cloud_model, cloud_data, epochs=10)  # fine-tuning
# cloud_acc_finetune1 = test(cloud_model, test_dataset)

print(">>> Start distillation from party B to cloud...")
# distill(nn.Sequential(feature_extractor, party_B_backbone), nn.Sequential(feature_extractor, cloud_backbone), party_B_train, party_B_val, loss_fn_kd)
distill(party_B_model, cloud_model, party_B_train, party_B_val, loss_fn_kd)
cloud_acc_kd2 = test(cloud_model, test_data)

# print(">>> Start fine-tuning cloud model...")
# train(cloud_model, cloud_data, epochs=10)  # fine-tuning
# cloud_acc_finetune2 = test(cloud_model, test_dataset)

print(f'Cloud model accuracy: {cloud_acc:.4f}')
print(f'Party A model accuracy: {party_A_acc:.4f}')
print(f'Party B model accuracy: {party_B_acc:.4f}')
print(f'Cloud model accuracy after distillation from party A: {cloud_acc_kd1:.4f}')
# print(f'Cloud model accuracy after fine-tuning: {cloud_acc_finetune1:.4f}')
print(f'Cloud model accuracy after distillation from party B: {cloud_acc_kd2:.4f}')
# print(f'Cloud model accuracy after fine-tuning: {cloud_acc_finetune2:.4f}')

# TODO:
# - Split out the first few layers and the last linear layer from the models
# 具体来说，将最前面的低层特征提取层和最后面的分类层分离出来
# - Add mutual distillation between the cloud and the parties
# - Add cloud data during fine-tuning to avoid catastrophic forgetting
