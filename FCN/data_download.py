

from torchvision.datasets import VOCSegmentation


voc_trainset = VOCSegmentation('./voc_seg',year='2012', image_set='train', download=True)
print(voc_trainset)