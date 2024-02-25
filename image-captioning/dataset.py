import albumentations as A
import cv2
import torch
from config import Config
from torch.utils.data import DataLoader


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        assert len(image_filenames) == len(captions), "Number of images and captions should be same"

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(self.captions, padding=True, return_tensors="pt", truncation=True)
        self.transforms = transforms

    def __getitem__(self, idx):
        # There are input_ids and attention_mask in the encoded_captions
        item = {key: val[idx] for key, val in self.encoded_captions.items()}

        image = cv2.imread(f"{Config.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        item["image"] = torch.tensor(image).permute(2, 0, 1).float()
        item["caption"] = self.captions[idx]  # only for visualization

        return item

    def __len__(self):
        return len(self.captions)

def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose([
            A.Resize(Config.size, Config.size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ])
    else:
        return A.Compose([
            A.Resize(Config.size, Config.size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ])

def parse_caption_file(caption_path):
    image_filenames = []
    captions = []
    with open(caption_path, "r") as f:
        for line in f:
            splited_line = line.strip().split("\t")
            image_filename = splited_line[0].split("#")[0]
            caption = splited_line[1]
            image_filenames.append(image_filename)
            captions.append(caption)
    return image_filenames, captions

def split_data(image_filenames, captions, train_size=0.8):
    train_size = int(len(image_filenames) * train_size)
    return (image_filenames[:train_size], captions[:train_size]), (image_filenames[train_size:], captions[train_size:])

def build_loader(image_filenames, captions, tokenizer=None, mode="train"):
    dataset = CLIPDataset(image_filenames, captions, tokenizer, get_transforms(mode=mode))
    dataloader = DataLoader(
        dataset, 
        batch_size=Config.batch_size, 
        shuffle=True if mode == "train" else False, 
        num_workers=Config.num_workers)
    return dataloader
