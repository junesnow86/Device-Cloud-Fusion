import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model,
    train_data,
    val_data,
    epochs=100,
    batch_size=128,
    lr=0.01,
    optimizer=None,
    device="cuda",
    patience=10,
    checkpoint_save_path=None,
):
    # warmup_epochs = int(epochs * 0.1)

    # def lr_lamba(current_epoch):
    #     if current_epoch < warmup_epochs:
    #         return (current_epoch / warmup_epochs) * lr
    #     return lr

    model.to(device)
    model.train()

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )

    # warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lamba)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_loss = float("inf")
    wait = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(
            train_dataloader, desc=f"Epoch {epoch+1} training", leave=False
        ):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        with torch.no_grad():
            val_loss = 0.0
            for inputs, labels in tqdm(
                val_dataloader, desc=f"Epoch {epoch+1} validation", leave=False
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # if epoch < warmup_epochs:
        #     warmup_scheduler.step()
        # else:
        #     scheduler.step()

        print(
            f"Epoch {epoch+1}: training loss={running_loss/len(train_dataloader)}, validation loss={val_loss/len(val_dataloader)}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            if checkpoint_save_path:
                model.to("cpu")
                torch.save(model.state_dict(), checkpoint_save_path)
                model.to(device)
        else:
            wait += 1
            if wait == patience:
                print(f"Early stop at epoch {epoch+1}.")
                break


def distill(
    student,
    teacher,
    transfer_data,
    val_data,
    temperature=1,
    alpha=0.5,
    epochs=100,
    batch_size=128,
    lr=0.01,
    optimizer=None,
    device="cuda",
    patience=5,
    checkpoint_save_path=None,
):
    """
    train_data: the data used to teach the student model
    """

    def soft_loss_fn(student_outputs, teacher_outputs, T=1):
        soft_loss = nn.KLDivLoss(reduction="batchmean")(
            nn.functional.log_softmax(student_outputs / T, dim=1),
            nn.functional.softmax(teacher_outputs / T, dim=1),
        ) * (T * T)
        return soft_loss

    teacher.to(device)
    student.to(device)
    teacher.eval()
    student.train()

    transfer_dataloader = DataLoader(transfer_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    hard_loss_fn = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = optim.SGD(
            student.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )

    best_loss = float("inf")
    wait = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(
            transfer_dataloader, desc="distillation training", leave=False
        ):
            # use teacher logits as soft targets
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            student_outputs = student(inputs)

            # calculate loss conditionally
            # _, teacher_preds = torch.max(teacher_outputs, 1)
            # loss = 0.0
            # for i in range(inputs.size(0)):
            #     hard_loss = hard_loss_fn(
            #         student_outputs[i].unsqueeze(0), labels[i].unsqueeze(0)
            #     )
            #     # If the teacher's prediction is correct, use soft loss
            #     if teacher_preds[i] == labels[i]:
            #         soft_loss = soft_loss_fn(
            #             student_outputs[i].unsqueeze(0),
            #             teacher_outputs[i].unsqueeze(0),
            #             T=temperature,
            #         )
            #         loss += (1 - alpha) * soft_loss + alpha * hard_loss
            #     else:
            #         loss += hard_loss
            # loss /= inputs.size(0)

            soft_loss = soft_loss_fn(student_outputs, teacher_outputs, T=temperature)
            hard_loss = hard_loss_fn(student_outputs, labels)
            loss = (1 - alpha) * soft_loss + alpha * hard_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(
                val_dataloader, desc="distillation validation", leave=False
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)
                student_outputs = student(inputs)
                loss = hard_loss_fn(student_outputs, labels)
                val_loss += loss.item()

        print(
            f"Epoch {epoch+1}: training loss={running_loss/len(transfer_dataloader)}, validation loss={val_loss/len(val_dataloader)}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            if checkpoint_save_path:
                student.to("cpu")
                torch.save(student.state_dict(), checkpoint_save_path)
                student.to(device)
        else:
            wait += 1
            if wait == patience:
                print(f"Early stop at epoch {epoch+1}.")
                break


def fed_avg(model_weights, sample_numbers):
    avg_weights = {}
    keys = model_weights[0].keys()

    for key in keys:
        layer_weights = [
            model_weight[key].clone().detach() * num
            for model_weight, num in zip(model_weights, sample_numbers)
        ]
        layer_weights_avg = sum(layer_weights) / sum(sample_numbers)
        avg_weights[key] = layer_weights_avg

    return avg_weights


@torch.no_grad()
def test_accuracy(model, test_data, batch_size=64, device="cuda"):
    model.to(device)
    model.eval()

    dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    total, correct = 0, 0

    for inputs, labels in tqdm(dataloader, desc="Testing", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        preds = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return correct / total


@torch.no_grad()
def test_accuracy_with_preds(model, test_data, batch_size=64, device="cuda"):
    model.to(device)
    model.eval()

    dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    total, correct = 0, 0

    for inputs, labels in tqdm(dataloader, desc="Testing", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        preds = model(inputs)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return correct / total
