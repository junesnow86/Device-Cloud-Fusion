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
    batch_size=64,
    lr=0.001,
    optimizer=None,
    device="cuda",
    patience=5,
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
                torch.save(model.state_dict(), checkpoint_save_path)
        else:
            wait += 1
            if wait == patience:
                print(f"Early stop at epoch {epoch+1}.")
                break
