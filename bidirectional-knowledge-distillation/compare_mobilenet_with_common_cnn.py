import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import mobilenet_v3_small, squeezenet1_1

from modules.data_utils import TinyImageNet200
from modules.functional import test_accuracy, train

seed = 100
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 定义数据预处理
transform = T.Compose(
    [
        T.ToImage(),
        T.ToDtype(torch.float32),
        # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        T.Normalize(
            (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
        ),  # TinyImageNet200数据集的均值和标准差
    ]
)

# train_data = CIFAR10(root="./data", train=True, transform=transform)
# test_data = CIFAR10(root="./data", train=False, transform=transform)
# train_data = CIFAR100(root="./data", train=True, transform=transform)
# test_data = CIFAR100(root="./data", train=False, transform=transform)
train_data = TinyImageNet200(type_="train", transform=transform)
test_data = TinyImageNet200(type_="val", transform=transform)

# 划分验证集
n_train = int(len(train_data) * 0.9)
n_validation = len(train_data) - n_train
train_data, valid_data = random_split(train_data, [n_train, n_validation])

# 定义SqueezeNet 1.1模型
# squeezenet = squeezenet1_1(weights=None, num_classes=200)
mobilenet = mobilenet_v3_small(weights=None, num_classes=200)


# 定义一个普通的CNN模型
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1
        )  # 修改stride为1
        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=3, stride=1, padding=1
        )  # 修改stride为1
        # self.fc1 = nn.Linear(128 * 8 * 8, 1024)  # 修改输入大小为128 * 8 * 8
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 200)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


mycnn = MyCNN()


# 计算模型参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / (1024**2)


# 打印参数数量
# print(f"SqueezeNet 1.1 parameters: {count_parameters(squeezenet)} MB")
print(f"MobileNetV3-Small parameters: {count_parameters(mobilenet)} MB")
print(f"My CNN parameters: {count_parameters(mycnn)} MB")

train(
    mycnn,
    train_data,
    valid_data,
    # checkpoint_save_path="./checkpoints/mycnn_tinyimagenet200.pth",
    checkpoint_save_path="./checkpoints/mycnn_cifar10.pth",
)
# mycnn.load_state_dict(torch.load("./checkpoints/mycnn_tinyimagenet200.pth"))
mycnn.load_state_dict(torch.load("./checkpoints/mycnn_cifar10.pth"))
accuracy_mycnn = test_accuracy(mycnn, test_data)

# train(
#     squeezenet,
#     train_data,
#     valid_data,
#     # lr=0.01,
#     checkpoint_save_path="./checkpoints/squeezenet_tinyimagenet200.pth",
# )
# squeezenet.load_state_dict(torch.load("./checkpoints/squeezenet_tinyimagenet200.pth"))
# accuracy_squeezenet = test_accuracy(squeezenet, test_data)

train(
    mobilenet,
    train_data,
    valid_data,
    # checkpoint_save_path="./checkpoints/mobilenet_tinyimagenet200.pth",
    checkpoint_save_path="./checkpoints/mobilenet_cifar10.pth",
)
# mobilenet.load_state_dict(torch.load("./checkpoints/mobilenet_tinyimagenet200.pth"))
mobilenet.load_state_dict(torch.load("./checkpoints/mobilenet_cifar10.pth"))
accuracy_mobilenet = test_accuracy(mobilenet, test_data)

# print(f"SqueezeNet accuracy: {accuracy_squeezenet}")
print(f"MobileNetV3-Small accuracy: {accuracy_mobilenet}")
print(f"My CNN accuracy: {accuracy_mycnn}")
