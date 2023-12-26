
import torch
import torchvision
from torch import nn


class Resnet50(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.net = torchvision.models.resnet50(pretrained=True)
        num_in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(num_in_features, 10)

    def forward(self, x):
        y = self.net(x)
        return y
