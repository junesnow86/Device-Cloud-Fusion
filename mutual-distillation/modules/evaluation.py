import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


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
