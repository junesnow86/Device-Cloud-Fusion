# Device-Cloud-Fusion

## Datasets for multimodal task

### Cross-Modal Federated Human Activity Recognition via Modality-Agnostic and Modality-Specific Representation Learning(AAAI2022)

**EPIC-KITCHENS-100**: EPIC-KITCHENS-100是一个大规模的自我为中心的人体行为识别数据集，主要用于研究厨房环境中的人与物体互动。以下是该数据集的一些重要特征和信息：

1. **规模和来源：** 数据集包含了来自37位参与者的89977个视频段，这些参与者通过穿戴式摄像头记录了在厨房环境中进行的各种活动。这使得该数据集成为目前最大的自我为中心的多模态数据集之一。

2. **视频内容：** 视频涵盖了各种与厨房相关的活动，例如烹饪、清理、取食物等。这些视频提供了大量的场景和动作多样性，使其成为研究自我为中心的人体行为识别问题的理想选择。

3. **人体与物体互动：** 数据集中的视频主要聚焦于人与物体的互动，包括拿、放、切、搅拌等各种动作。这使得该数据集适用于人体行为识别、物体识别以及二者的联合任务。

4. **多模态数据：** 除了视频数据外，EPIC-KITCHENS-100还包含了一些参与者提供的音频和传感器数据。这提供了一个丰富的多模态观测视角，有助于研究在不同模态下的人体行为识别问题。

5. **标签：** 数据集使用了97个动作标签，涵盖了各种厨房活动。这些标签在研究中被用作活动类别的基础，例如在上述提到的CMF-HAR任务中。


### Multimodal Federated Learning via Contrastive Representation Ensemble(CreamFL, ICLR2023)

具有 50,000 个图像文本对的 COCO （Lin et al.， 2014） 的随机子集被选为公共多模态数据。

CIFAR100、AG NEWS 和 Flicker30k 分别用作图像、文本和多模态客户端的私有数据集。

1. CIFAR-100（Krizhevsky等人，2009）由100个类的50,000个彩色训练图像组成，每个类有500个图像。

2. AG NEWS（Zhang et al.， 2015）包含来自 4 个类的 120,000 个训练句子。

3. Flicker30k（Plummer 等人，2015 年）包含从 Flicker 收集的 31,000 张图像，以及每张图像 5 个标题，即总共 155,000 个图像文本对。

对于跨模态检索，我们遵循Karpathy&Fei-Fei（2015）并报告了 MS-COCO 的5K / 1K测试集的Recall@K结果，即测量了在前K结果中找到正确 item 的次数百分比。

对于视觉问答，我们使用 VQA v2.0 数据集（Goyal 等人，2017 年）并报告 3,000 个最常见答案的准确性。


### DreamLLM: Synergistic Multimodal Comprehension and Creation

#### Multimodal Comprehension

- image-to-text captioning on COCO

- general visual question answering (VQA) on VQAv2 

#### Text-conditional Image Synthesis

MS COCO, LN COCO, 使用 zero-shot Frechet Inception Distance(FID) 作为评价指标。

#### Multimodal Joint Creation and Comprehension

(omit)