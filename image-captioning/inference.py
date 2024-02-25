import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from config import Config
from dataset import build_loader
from model import CLIPModel
from tqdm import tqdm
from transformers import DistilBertTokenizer


@torch.no_grad()
def get_image_embeddings(valid_image_filenames, valid_captions, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(Config.text_tokenizer)
    valid_loader = build_loader(valid_image_filenames, valid_captions, tokenizer, mode="valid")

    model = CLIPModel().to(Config.device)
    model.load_state_dict(torch.load(model_path, map_location=Config.device))
    model.eval()

    valid_image_embeddings = []
    for batch in tqdm(valid_loader, total=len(valid_loader), desc="Getting image embeddings"):
        image_features = model.image_encoder(batch["image"].to(Config.device))
        image_embeddings = model.image_projection(image_features)
        valid_image_embeddings.append(image_embeddings)

    return model, torch.cat(valid_image_embeddings)

@torch.no_grad()
def find_matches(model, image_embeddings, query, image_filenames, k=9):
    tokenizer = DistilBertTokenizer.from_pretrained(Config.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {key: val.to(Config.device) for key, val in encoded_query.items()}
    text_features = model.text_encoder(batch["input_ids"], batch["attention_mask"])
    text_embeddings = model.text_projection(text_features)

    image_embeddings_k = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_k = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_k @ image_embeddings_k.T

    values, indices = torch.topk(dot_similarity.squeeze(0), k * 5)
    matches = [image_filenames[i] for i in indices[::5]]

    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{Config.image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")

    plt.savefig("matches.png")
