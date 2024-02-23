
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions, lengths):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs

# 参数设置
embed_size = 256
hidden_size = 512
vocab = ["<start>", "a", "an", "the", "on", "in", "near", "under", "is", "are"]
vocab_size = len(vocab)
num_layers = 1

# 实例化模型
model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Flickr30k

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 加载数据集
train_dataset = Flickr30k(root_dir='path/to/flickr30k',
                                 anno_file='path/to/annotations.txt',
                                 transform=transform)

val_dataset = Flickr30k(root_dir='path/to/flickr30k',
                               anno_file='path/to/val_annotations.txt',
                               transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, captions, lengths in train_loader:
        optimizer.zero_grad()
        outputs = model(images, captions, lengths)
        loss = criterion(outputs, captions.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_dataset)
    print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss))

    # 验证模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, captions, lengths in val_loader:
            outputs = model(images, captions, lengths)
            loss = criterion(outputs, captions.view(-1))
            val_loss += loss.item() * images.size(0)

    val_loss = val_loss / len(val_dataset)
    print('Epoch [{}/{}], Val Loss: {:.4f}'.format(epoch+1, num_epochs, val_loss))

# 保存模型
torch.save(model.state_dict(), 'image_captioning_model.pth')

# 加载模型
model.load_state_dict(torch.load('image_captioning_model.pth'))

# 测试模型
def predict_caption(image, max_length=20):
    model.eval()
    with torch.no_grad():
        features = model.encoder(image.unsqueeze(0))
        states = None
        input = torch.tensor([vocab['<start>']]).unsqueeze(0)
        caption = []

        for _ in range(max_length):
            hiddens, states = model.decoder.lstm(input, states)
            output = model.decoder.linear(hiddens.squeeze(1))
            predicted = output.argmax(1)
            caption.append(predicted.item())
            input = predicted.unsqueeze(0)

            if vocab.idx2word[predicted.item()] == '':
                break

        caption = [vocab.idx2word[idx] for idx in caption]
        return ' '.join(caption)

# 测试一张图片
from PIL import Image
image_path = 'path/to/test_image.jpg'
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0)
caption = predict_caption(image)
print('Generated Caption:', caption)
