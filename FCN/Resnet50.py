
import math

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, down_sample=None, dilation_rate=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3,
                               stride=stride, padding=dilation_rate, bias=False, dilation=dilation_rate)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*4)

        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample

    def forward(self, x):
        identity = x

        conv1_out = self.conv1(x)
        bn1_out = self.bn1(conv1_out)
        relu1_out = self.relu(bn1_out)

        conv2_out = self.conv2(relu1_out)
        bn2_out = self.bn2(conv2_out)
        relu2_out = self.relu(bn2_out)

        conv3_out = self.conv3(relu2_out)
        bn3_out = self.bn3(conv3_out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        bn3_out += identity

        out = self.relu(bn3_out)

        return out


class Resnet(nn.Module):
    def __init__(self, bottleneck_num, num_classes, replace_conv=None):
        """
        构建残差网络
        :param bottleneck_num: 残差块数量
        :param num_classes: 分类数量
        :param replace_conv:  是否使用膨胀卷积
        """
        super().__init__()
        if replace_conv is None:
            replace_conv = [False, False, False]

        self.in_channel = 64
        self.dilation_rate = 1

        if len(replace_conv) != 3:
            raise ValueError(f"replace stride with dilation should be None or a 3-element tuple, got {replace_conv}")

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 64, 128, 256, 512 分别是 每一层第一个block里第一个convolution的out_channel
        self.layer1 = self._make_block(64, bottleneck_num[0])
        self.layer2 = self._make_block(128, bottleneck_num[1], stride=2, replace_conv=replace_conv[0])
        self.layer3 = self._make_block(256, bottleneck_num[2], stride=2, replace_conv=replace_conv[1])
        self.layer4 = self._make_block(512, bottleneck_num[3], stride=2, replace_conv=replace_conv[2])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.bn1(out1)
        out3 = self.relu(out2)
        out4 = self.max_pool(out3)

        out5 = self.layer1(out4)
        out6 = self.layer2(out5)
        out7 = self.layer3(out6)
        out8 = self.layer4(out7)

        out9 = self.avg_pool(out8)
        out10 = torch.flatten(out9, 1)
        out = self.fc(out10)

        return out

    def _make_block(self, out_channel, block_num, stride=1, replace_conv=False):
        down_sample = None
        previous_dilation_rate = self.dilation_rate
        if replace_conv:
            self.dilation_rate *= stride
            stride = 1

        # 每一个layer的第一个block, down_sample表示跨层连接，是否需要下采样
        if stride != 1 or self.in_channel != out_channel * 4:
            down_sample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=out_channel*4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel*4)
            )

        layers = [Bottleneck(self.in_channel, out_channel, stride, down_sample, previous_dilation_rate)]
        self.in_channel = out_channel * 4

        for _ in range(1, block_num):
            layers.append(Bottleneck(self.in_channel, out_channel, dilation_rate=self.dilation_rate))

        return nn.Sequential(*layers)


def resnet50(**kwargs):
    model = Resnet([3, 4, 6, 3], num_classes=21, **kwargs)
    return model
