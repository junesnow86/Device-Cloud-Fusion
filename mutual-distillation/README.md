## Implementation Steps
- [x] train a custom ResNet-18 on CIFAR-10
- [x] split FashionMNIST into 3 random parts
- [x] distill 3 small models with custom ResNet-18 on CIFAR-10
- [x] fine-tune 3 small models on local FashionMNIST sub-dataset
- [x] add common classifier layer, one classifer for each dataset(to prove that small models learn useful knowledge from local datasets)
- [x] compare performance of fine-tuned small models' ensemble and separate small models
- [ ] distill small models to ResNet-18 with model ensemble
- [ ] distill ResNet-18 to small models

## Q & A

Q1. 端设备的数据集具体该如何划分？

A1. 1）共用一个测试集，还是不同参与方分到不同的测试集？共用一个测试集，可以表现模型在用户数据全集上的整体性能，个性化评测可以使用验证集。2）先将整个数据集划分成训练集和测试集，然后将训练集划分成三个部分，分配给每个参与方，然后再在每个参与方训练集上划分出训练集和验证集。

Q2. 为什么换用CIFAR-10和FashionMNIST？

A2. Food-101数据量很大，训练一次需要很长时间，换用更小更容易训练的数据集来尽快完成预实验。

Q3. baseline怎么确定？