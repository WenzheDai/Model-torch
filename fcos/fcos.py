
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn


class FCOS(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass