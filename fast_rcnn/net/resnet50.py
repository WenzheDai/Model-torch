import torchvision
from torch import nn


def resnet50():
    model = torchvision.models.resnet50()
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])

    classifier = list([model.layer4, model.avgpool])

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)

    return features, classifier
