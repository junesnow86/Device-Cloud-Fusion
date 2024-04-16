from torchvision.models import (
    mobilenet_v3_small,
    resnet18,
    shufflenet_v2_x1_0,
    squeezenet1_0,
)

print(resnet18(weights=None, num_classes=10))
print(mobilenet_v3_small(weights=None, num_classes=10))
print(shufflenet_v2_x1_0(weights=None, num_classes=10))
print(squeezenet1_0(weights=None, num_classes=10))
