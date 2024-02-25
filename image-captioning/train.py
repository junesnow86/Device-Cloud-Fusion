import itertools

import torch
from config import Config
from dataset import build_loader, parse_caption_file, split_data
from model import CLIPModel
from tqdm import tqdm
from transformers import DistilBertTokenizer
from utils import AvgMeter, get_lr


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    progress_bar = tqdm(train_loader, total=len(train_loader), desc="training")
    for batch in progress_bar:
        batch = {k: v.to(Config.device) for k, v in batch.items() if k != "caption"}
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()

        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        progress_bar.set_postfix(loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter

@torch.no_grad()
def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    progress_bar = tqdm(valid_loader, total=len(valid_loader), desc="validation")
    for batch in progress_bar:
        batch = {k: v.to(Config.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        progress_bar.set_postfix(loss=loss_meter.avg)

    return loss_meter

def main():
    import time
    print("Loading data...")
    start = time.time()
    image_filenames, captions = parse_caption_file(Config.caption_path)
    train_data, valid_data = split_data(image_filenames, captions)
    tokenizer = DistilBertTokenizer.from_pretrained(Config.text_tokenizer)
    train_loader = build_loader(*train_data, tokenizer=tokenizer, mode="train")
    valid_loader = build_loader(*valid_data, tokenizer=tokenizer, mode="valid")
    print(f"Data loaded in {time.time() - start:.2f} seconds")

    model = CLIPModel().to(Config.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": Config.image_enoder_lr},
        {"params": model.text_encoder.parameters(), "lr": Config.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(),
            model.text_projection.parameters()
        ), "lr": Config.head_lr, "weight_decay": Config.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=Config.patience, factor=Config.factor, verbose=True
    )
    step = "epoch"

    best_loss = float("inf")
    for epoch in range(Config.epochs):
        print(f"Epoch {epoch + 1}/{Config.epochs}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        valid_loss = valid_epoch(model, valid_loader)

        lr_scheduler.step(valid_loss.avg)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best_model.pth")
            print("best model saved")

        print(f"epoch train loss: {train_loss.avg:.4f}, valid loss: {valid_loss.avg:.4f}")

if __name__ == "__main__":
    main()
