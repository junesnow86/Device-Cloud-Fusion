import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model,
    train_data,
    val_data,
    epochs=50,
    batch_size=32,
    lr=0.001,
    optimizer=None,
    device="cuda",
    patience=5,
    save_path=None,
):
    model.to(device)
    model.train()
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    best_loss = float("inf")
    wait = 0
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
            wait = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait == patience:
                print(f"Early stop at epoch {epoch+1}.")
                break


def test(model, data, batch_size=32, device="cuda"):
    model.to(device)
    model.eval()
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def distill(
    teacher,
    student,
    transfer_data,
    val_data,
    epochs=10,
    batch_size=32,
    lr=0.001,
    optimizer=None,
    device="cuda",
    patience=5,
    save_path=None,
):
    """
    train_data: the data used to teach the student model
    """

    def loss_fn_kd(student_outputs, teacher_outputs, labels=None, T=1, alpha=0.2):
        soft_loss = nn.KLDivLoss(reduction="batchmean")(
            nn.functional.log_softmax(student_outputs / T, dim=1),
            nn.functional.softmax(teacher_outputs / T, dim=1),
        ) * (T * T)
        if labels is not None:
            hard_loss = nn.CrossEntropyLoss()(student_outputs, labels)
            soft_loss = soft_loss * alpha + hard_loss * (1 - alpha)
        return soft_loss

    teacher.to(device)
    student.to(device)
    teacher.eval()
    # Freeze teacher's parameters
    for param in teacher.parameters():
        param.requires_grad = False
    student.train()
    transfer_dataloader = DataLoader(transfer_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    val_loss_fn = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9)

    best_loss = float("inf")
    wait = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(transfer_dataloader, desc="distillation training"):
            # use teacher logits as soft targets
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            student_outputs = student(inputs)
            loss = loss_fn_kd(student_outputs, teacher_outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_dataloader, desc="distillation validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                student_outputs = student(inputs)
                loss = val_loss_fn(student_outputs, labels)
                val_loss += loss.item()

        print(
            f"Epoch {epoch+1}: training loss={running_loss/len(transfer_dataloader)}, validation loss={val_loss/len(val_dataloader)}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            if save_path:
                torch.save(student.state_dict(), save_path)
        else:
            wait += 1
            if wait == patience:
                print(f"Early stop at epoch {epoch+1}.")
                break


def calculate_model_size(model):
    return (
        sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
        + sum(p.numel() * p.element_size() for p in model.buffers())
    ) / (1024**2)


if __name__ == "__main__":
    from torchvision.models import (
        mobilenet_v3_large,
        mobilenet_v3_small,
        resnet18,
        resnet34,
        resnet50,
        resnet152,
        shufflenet_v2_x1_0,
        squeezenet1_0,
    )

    model = resnet152(weights=None, num_classes=10)
    print(calculate_model_size(model))
    model = resnet50(weights=None, num_classes=10)
    print(calculate_model_size(model))
    model = resnet34(weights=None, num_classes=10)
    print(calculate_model_size(model))
    model = resnet18(weights=None, num_classes=10)
    print(calculate_model_size(model))
    model = mobilenet_v3_large(weights=None, num_classes=10)
    print(calculate_model_size(model))
    model = mobilenet_v3_small(weights=None, num_classes=10)
    print(calculate_model_size(model))
    model = shufflenet_v2_x1_0(weights=None, num_classes=10)
    print(calculate_model_size(model))
    model = squeezenet1_0(weights=None, num_classes=10)
    print(calculate_model_size(model))
