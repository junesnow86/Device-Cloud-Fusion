import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.models import mobilenet_v2, resnet18, resnet152
from tqdm import tqdm


def loss_fn_kd(outputs, teacher_outputs, labels=None, T=20, alpha=0.5, with_labels=False):
    soft_loss = nn.KLDivLoss(reduction="batchmean")(nn.functional.log_softmax(outputs/T, dim=1),
                               nn.functional.softmax(teacher_outputs/T, dim=1)) * (T * T)
    if with_labels:
        hard_loss = nn.CrossEntropyLoss()(outputs, labels) * (1. - alpha)
        soft_loss = soft_loss * alpha + hard_loss
    return soft_loss

def train(model, data, epochs=20, device='cuda'):
    model.to(device)
    model.train()
    dataloader = DataLoader(data, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}: Loss = {running_loss/len(dataloader)}')

def test(model, data, device='cuda'):
    model.to(device)
    model.eval()
    dataloader = DataLoader(data, batch_size=32, shuffle=False)
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

def distill(teacher, student, data, loss_fn, epochs=10, device='cuda'):
    teacher.to(device)
    student.to(device)
    teacher.eval()
    student.train()
    dataloader = DataLoader(data, batch_size=32, shuffle=True)
    optimizer = optim.SGD(student.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            teacher_outputs = teacher(inputs)
            student_outputs = student(inputs)
            loss = loss_fn(student_outputs, teacher_outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}: Loss = {running_loss/len(dataloader)}')

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split dataset according to the class
class_indices = [np.where(np.array(train_dataset.targets) == i)[0] for i in range(10)]
cloud_indices = np.concatenate(class_indices[:2])
party_A_indices = np.concatenate(class_indices[2:6])
party_B_indices = np.concatenate(class_indices[6:])
cloud_data = Subset(train_dataset, cloud_indices)
party_A_data = Subset(train_dataset, party_A_indices)
party_B_data = Subset(train_dataset, party_B_indices)

# Define models
cloud_model = resnet152(weights=None, num_classes=10)
party_A_model = resnet18(weights=None, num_classes=10)
party_B_model = mobilenet_v2(weights=None, num_classes=10)

# Train models respectively with the splitted dataset
print(">>> Start training cloud model...")
train(cloud_model, cloud_data)
cloud_acc = test(cloud_model, test_dataset)

print(">>> Start training party A model...")
train(party_A_model, party_A_data)
party_A_acc = test(party_A_model, test_dataset)

print(">>> Start training party B model...")
train(party_B_model, party_B_data)
party_B_acc = test(party_B_model, test_dataset)

print(f'Cloud model accuracy: {cloud_acc:.4f}')
print(f'Party A model accuracy: {party_A_acc:.4f}')
print(f'Party B model accuracy: {party_B_acc:.4f}')

# Perform knowledge distillation from parties to the cloud
print(">>> Start distillation from party A to cloud...")
distill(party_A_model, cloud_model, party_A_data, loss_fn_kd)
cloud_acc_kd1 = test(cloud_model, test_dataset)

print(">>> Start distillation from party B to cloud...")
distill(party_B_model, cloud_model, party_B_data, loss_fn_kd)
cloud_acc_kd2 = test(cloud_model, test_dataset)

print(f'Cloud model accuracy after distillation from party A: {cloud_acc_kd1:.4f}')
print(f'Cloud model accuracy after distillation from party B: {cloud_acc_kd2:.4f}')

# TODO:
# - Add an embedding layer before the CNN models
# - Add fine-tuning process after distillation
