## 2024.3.24
根据 main03241046.log 和 main03241258.log 的结果，双向蒸馏在端侧模型上取得了一定的效果，三个参与方中有两个参与方的性能得到了提高，但是一个参与方的性能下降了。
而有一个很奇怪的点是云侧模型在蒸馏后准确率降为 0.0，这个问题需要进一步排查。

## 2024.3.25
后面打算在设备到云的蒸馏时考虑数据筛选，这样的话分别使用 CIFAR-10 和 Caltech-101 作为设备数据集和云数据集就不自然。打算使用 ImageNet，划分成云数据集和设备数据集。
还有前面写的代码里，从云到设备的蒸馏是不对的，因为云模型不能直接下载到端。而且蒸馏还写成调用`train`了。。。

## 2024.4.10
云端数据集使用的是CIFAR-10，设备端数据集使用的是FashionMNIST，在进行设备端本地微调后，设备端的三个小模型在云端测试集上表现为10%左右的准确率（相当于随机选），而在进行本地微调之前，准确率有70%以上。
这是可以理解的，因为设备端小模型的分类器层对应的标签语义是FashionMNIST的10个标签。

而经过实验验证，在ResNet-18黑盒的基础上，**外接一层线性层分类器**，前后性能表现相差无几（0.8005 v.s. 0.7909）。
因此，可以考虑添加外置的dataset-specific classifier来尝试解决上述问题。

custom cloud resnet18 pretrained: 0.8025
custom distill mobilenet pretrained: 0.7225
custom distill shufflenet pretrained: 0.7234
custom distill squeeze pretrained: 0.7031

## 2024.4.14
pretrained custom mobilenet test on FashionMNIST: 0.1112
fine-tuned custom mobilenet test on FashionMNIST: 0.7809

pretrained custom shufflenet test on FashionMNIST: 0.1004
fine-tuned custom shufflenet test on FashionMNIST: 0.8526

pretrained custom squeezenet test on FashionMNIST: 0.1110
fine-tuned custom squeezenet test on FashionMNIST: 0.8672

---------------------------------------------------------
使用fine-tuned backbone + pretrained classifier在CIFAR-10上进行测试，准确率只有0.1，也就是说**保留pretrained classifier没有作用**。
**微调后的backbone起决定性作用，会导致小模型遗忘预训练数据集的特征提取能力。**
我认为这种情况下，使用小模型蒸馏，对大模型增益不会有积极作用。
但是当初为什么这样做呢？是因为看到了一篇arbitrary transfer set的论文，受到了其影响。
---------------------------------------------------------
验证ensemble的提升效果：
pretrained
Model 0 accuracy: 0.7292                                                                                                                                                                                                                                
Model 1 accuracy: 0.7289
Model 2 accuracy: 0.7276
Ensemble accuracy: 0.7855
Ensemble vote accuracy: 0.7620

fine-tuned
Model 0 fine-tune accuracy: 0.8048                                                                                                                                                                                                                      
Model 1 fine-tune accuracy: 0.8608
Model 2 fine-tune accuracy: 0.8704
Ensemble fine-tune accuracy: 0.8875
Ensemble vote fine-tune accuracy: 0.8829

## 2024.4.15
Cloud model accuracy: 0.4411                                                                                                                                                                                             
Device model 0 accuracy: 0.4904
Device model 1 accuracy: 0.5397
Device model 2 accuracy: 0.5258
Device ensemble accuracy: 0.5857
Cloud model accuracy after distillation: 0.4991
ensemble -> cloud 蒸馏后的云端模型是有了性能提升，但是并没有超越任何一个device小模型，更不用说超越ensemble了。
这样的话就会使得second KD没有意义。
而盲目地堆叠数据增强操作甚至会令蒸馏后的性能低于原始模型。

## 2024.4.16
使用预训练权重和不使用太复杂的数据转换操作可以带来近20%%的性能提升。
resnet18 on CIFAR-10 10% with pretrained weights: 0.6305

较为理想的情况：
Cloud model accuracy: 0.6305                                         
Device model 0 accuracy: 0.5854
Device model 1 accuracy: 0.5957
Device model 2 accuracy: 0.5332
Device ensemble accuracy: 0.6445
Cloud model accuracy after distillation: 0.6976
在使用本地数据微调后的小模型ensemble性能超越了云端模型，而在蒸馏后，云端模型的性能超越了ensemble。

second distillation results
Cloud model accuracy before distillation: 0.6305
Device model 0 accuracy before second-distillation: 0.5854
Device model 1 accuracy before second-distillation: 0.5957
Device model 2 accuracy before second-distillation: 0.5332
Device ensemble accuracy before second-distillation: 0.6445
Cloud model accuracy: 0.6896
Device model 0 accuracy: 0.5918
Device model 1 accuracy: 0.5961
Device model 2 accuracy: 0.5481
Device ensemble accuracy: 0.6386
在经过第二次蒸馏后，每个小模型的性能都有微小的提升，但是ensemble的性能下降了。

squeezenet FedAvg: 使用最小的模型进行FedAvg聚合的性能表现要比大小异构模型的ensemble要好。
Round 0: 0.4227
Round 1: 0.5622
Round 2: 0.598
Round 3: 0.6225
Round 4: 0.6307
Round 5: 0.6311
Round 6: 0.6429
Round 7: 0.6447
Round 8: 0.643
Round 9: 0.6483
Round 10: 0.6503
Round 11: 0.6537
Round 12: 0.6527
Round 13: 0.657
Round 14: 0.6563
Round 15: 0.6535
Round 16: 0.6627
Round 17: 0.6606
Round 18: 0.6572
Round 19: 0.6569
Round 20: 0.6599

## 2024.4.18
将云端数据量增加到25%
Cloud model accuracy: 0.7741
Device model 0 accuracy: 0.6726
Device model 1 accuracy: 0.7135
Device model 2 accuracy: 0.6870
Device ensemble accuracy: 0.7523
Cloud model accuracy after distillation: 0.8039

二次蒸馏的设备模型性能提升十分有限：
mobilenet
Cloud model accuracy: 0.8004
Device model accuracy before distillation: 0.6726
Device model accuracy after distillation: 0.6796

shufflenet
Cloud model accuracy: 0.8004
Device model accuracy before distillation: 0.7135
Device model accuracy after distillation: 0.7197

squeezenet
Cloud model accuracy: 0.8004
Device model accuracy before distillation: 0.6870
Device model accuracy after distillation: 0.6899

Model 0 fine-tune accuracy: 0.6786
Model 1 fine-tune accuracy: 0.7202
Model 2 fine-tune accuracy: 0.6918
Ensemble fine-tune accuracy: 0.7474 （比second-distillation前低）
Ensemble vote fine-tune accuracy: 0.7318

squeezenet FedAvg:
Round 0: 0.6543
Round 1: 0.6869
Round 2: 0.6973
Round 3: 0.7045
Round 4: 0.7094
Round 5: 0.7102
Round 6: 0.7139
Round 7: 0.7169
Round 8: 0.7187
Round 9: 0.718
Round 10: 0.7266
Round 11: 0.7199
Round 12: 0.7248
Round 13: 0.7275
Round 14: 0.7264
Round 15: 0.7331
Round 16: 0.7353
Round 17: 0.7346
Round 18: 0.7345
Round 19: 0.739
Round 20: 0.7421

当将云端数据比例增加到40%时，端侧就不会带来增益了。
Cloud model accuracy: 0.8002
Device model 0 accuracy: 0.6962
Device model 1 accuracy: 0.6894
Device model 2 accuracy: 0.6952
Device ensemble accuracy: 0.7397
Cloud model accuracy after distillation: 0.8043

## 2024.4.19
mobilenet_v3_large trained on CIFAR-100, acc = 0.35
squeezenet_1_1 trained on CIFAR-100, acc = 0.32

mobilenet_v3_large trained on CIFAR-10, acc = 0.6759
squeezenet_1_1 trained on CIFAR-10, acc = 0.6941

pretrained mobilenet_v3_large fine-tuned on CIFAR-10, acc = 0.5599
pretrained squeezenet_1_1 fine-tuned on CIFAR-10, acc = 0.7036

mobilenet_v3_large 的模型参数量比 squeezenet_1_1 大，但是在相同条件下在 CIFAR-10 上的准确率不如 squeezenet_1_1。
这个实验、这个数据集不能体现端侧模型参数量所带来的差异。

## 2024.4.22
CIFAR-10, 10% for cloud resnet18, 90% for device mobilenet_v3_small
Accuracy of cloud model: 0.6894
Accuracy of device model: 0.6583
Accuracy of distilled cloud model: 0.7263

Observation:
1. 虽然 device model 在更多的数据上训练，但是在全局测试数据上的准确率还是不如 cloud model。
2. 虽然 device model 在全局测试数据上的准确率不如 cloud model，但是使用 device model 作为教师模型，在 cloud data 上进行知识蒸馏，还是能提高 cloud model 的准确率。

## 2024.4.23
尝试挖掘 model ensemble 和 FedAvg 之间性能差异的原因

squeezenet_1_1 trained with 10-clients fedavg:
Round 0: 0.101
Round 1: 0.2823
Round 2: 0.3323
Round 3: 0.3597
Round 4: 0.3664
Round 5: 0.384
Round 6: 0.3845
Round 7: 0.4042
Round 8: 0.4054
Round 9: 0.4233
Round 10: 0.4271
Round 11: 0.438
Round 12: 0.4425
Round 13: 0.4431
Round 14: 0.4505
Round 15: 0.4583
Round 16: 0.4584
Round 17: 0.4677
Round 18: 0.4681
Round 19: 0.4691
Round 20: 0.4612

Accuracy list: [0.2773, 0.1704, 0.2087, 0.2121, 0.2418, 0.2246, 0.2374, 0.2169, 0.2261, 0.2865]
Ensemble accuracy: 0.3906
FedAvg accuracy: 0.1

Round 0: 0.1
Round 1: 0.1
Round 2: 0.1625
Round 3: 0.328

## 2024.4.24
squeezenet训练不稳定，经常会出现loss不下降的现象，重新运行可能可以解决。

SqueezeNet accuracy: 0.4274
MobileNetV3-Small accuracy: 0.6211
My CNN accuracy: 0.7294

## 2024.4.25
on CIFAR-10
MobileNetV3-Small parameters: 1.457315444946289 MB
My CNN parameters: 8.082895278930664 MB
MobileNetV3-Small accuracy: 0.609
My CNN accuracy: 0.5228

on CIFAR-100
MobileNetV3-Small parameters: 1.5452919006347656 MB
My CNN parameters: 8.17087173461914 MB
MobileNetV3-Small accuracy: 0.3007
My CNN accuracy: 0.1347

on Tiny-ImageNet-200
MobileNetV3-Small parameters: 1.6430435180664062 MB
My CNN parameters: 32.26862335205078 MB
MobileNetV3-Small accuracy: 0.265
My CNN accuracy: 0.0051

这三组数据证明使用结构异构的一个潜在的好处，就是类似 MobileNet 的架构是做了同时考虑缩小参数量和保持性能的优化。

##
