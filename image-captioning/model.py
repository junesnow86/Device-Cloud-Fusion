import timm
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from transformers import DistilBertConfig, DistilBertModel


class ImageEncoder(nn.Module):
    """
    Encodes each image to a fixed size vector 
    with the size of the model's output channels
    (in case of ResNet50 the vector size will be 2048).
    This is the output after the nn.AdaptiveAvgPool2d() layer.
    """
    def __init__(self, model_name=Config.model_name, 
                 pretrained=Config.pretrained, 
                 trainable=Config.trainable) -> None:
        super().__init__()
        self.model = timm.create_model(model_name, 
                                       pretrained, 
                                       num_classes=0,
                                       global_pool="avg")
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    """
    In the case of DistilBERT (and also BERT) 
    the output hidden representation for each token is a vector with size 768. 
    So, the whole caption will be encoded in the CLS token representation whose size is 768.
    """
    def __init__(self, model_name=Config.text_encoder_model, 
                 pretrained=Config.pretrained, 
                 trainable=Config.trainable) -> None:
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig)

        for p in self.model.parameters():
            p.requires_grad = trainable

        #using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    """
    The projection head are simple linear layers that projects the 
    encoded image and text to a common space. 
    The output of this layer will be used to calculate the similarity score.
    """
    def __init__(self, 
                 embedding_dim,
                 projection_dim=Config.projection_dim,
                 dropout=Config.dropout) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        x = self.layer_norm(x)
        return x

class CLIPModel(nn.Module):
    def __init__(self,
                 temperature=Config.temperature,
                 image_embedding=Config.image_embedding,
                 text_embedding=Config.text_embedding) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(image_embedding)
        self.text_projection = ProjectionHead(text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        """
        First encode the images and texts separately into fixed size vectors
        (with different dimensionalities).
        Then, using separate projection modules to project them to the shared space.
        """
        # Getting image and text features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(input_ids=batch["input_ids"], 
                                          attention_mask=batch["attention_mask"])

        # Getting image and text embeddings(with the same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        image_similarity = image_embeddings @ image_embeddings.T
        text_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (image_similarity + text_similarity) / (2 * self.temperature), 
            dim=-1
        )
        text_loss_mean = nn.CrossEntropyLoss()(logits, targets)
        image_loss_mean = nn.CrossEntropyLoss()(logits.T, targets.T)
        loss_mean = (text_loss_mean + image_loss_mean) / 2.0  # shape: (batch size)
        return loss_mean  # shape: (1)
