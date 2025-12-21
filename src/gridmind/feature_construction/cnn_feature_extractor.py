import torch.nn as nn
from torchvision.models import resnet18


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim=512):  # 512 for resnet18, 2048 for resnet50
        super().__init__()
        resnet = resnet18(pretrained=True)
        self.features = nn.Sequential(
            *list(resnet.children())[:-2]
        )  # Remove avgpool & fc
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.flatten = nn.Flatten()
        self.output_dim = output_dim

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x  # shape: (B, output_dim)
