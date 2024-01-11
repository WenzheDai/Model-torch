

import torchvision
from torch import nn


class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = torchvision.models.resnet18()
        fc_input_feature = self.net.fc.in_features
        self.net.fc = nn.Linear(fc_input_feature, 2)

    def forward(self, x):
        return self.net(x)
