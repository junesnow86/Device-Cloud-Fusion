import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def distill(
    student,
    teacher,
    transfer_data,
    val_data,
    temperature=1,
    alpha=0.5,
    epochs=100,
    batch_size=32,
    lr=0.001,
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
                torch.save(student.state_dict(), checkpoint_save_path)
        else:
            wait += 1
            if wait == patience:
                print(f"Early stop at epoch {epoch+1}.")
                break
