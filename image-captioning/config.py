import torch


class Config:
    debug = False
    image_path = "/storage/1008ljt/Device-Cloud-Fusion/data/flickr30k/images/flickr30k-images"
    caption_path = "/storage/1008ljt/Device-Cloud-Fusion/data/flickr30k/captions/results_20130124.token"
    batch_size = 32
    num_workers = 4
    head_lr = 1e-3
    image_enoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "resnet50"
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True  # for both image encoder and text encoder
    trainable = True  # for both image encoder and text encoder
    temperature = 1.0

    size = 224  # image size

    # for projection head; uesd for both image and text
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1
