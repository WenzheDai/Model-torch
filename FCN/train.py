
import torch
import torch.nn.functional as F
import albumentations as A
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from fcn import fcn_resnet50
from dataset import VOCSegDataset


def criterion(predict, target):
    losses = {}
    for name, x in predict.items():
        losses[name] = F.cross_entropy(x, target)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


# def creator_lr_schedular(optimizer, num_step, epochs, warmup=True, warmup_epochs=1, warmup_factor=1e-3):
#     assert num_step > 0 and epochs > 0
#     if warmup is False:
#         warmup_epochs=0
#
#     def f(x):
#         if warmup is True and x <= (warmup_epochs * num_step):
#             alpha = float(x) / (warmup_epochs * num_step)
#             return warmup_factor * (1 - alpha) + alpha
#         else:
#             return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
#
#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


model = fcn_resnet50('aux')
model.cuda()

optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=0.0001,
                             weight_decay=0.0001)

# lr_schedular = creator_lr_schedular(optimizer, len(train_loader), epochs, warmup=True)
lr_schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

# argumentation
train_transform = A.Compose([
    A.Resize(224, 224),
    # A.RandomCrop(100, 100, p=0.5),
    # A.VerticalFlip(p=0.5),
    # A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

img_path = r"D:\git-work\Model-torch\FCN\voc_seg\VOCdevkit\VOC2012\JPEGImages"
gt_img_path = r"D:\git-work\Model-torch\FCN\voc_seg\VOCdevkit\VOC2012\SegmentationClass"

train_txt = r"D:\git-work\Model-torch\FCN\voc_seg\VOCdevkit\VOC2012\ImageSets\Segmentation\train.txt"
val_txt = r"D:\git-work\Model-torch\FCN\voc_seg\VOCdevkit\VOC2012\ImageSets\Segmentation\val.txt"

train_data = VOCSegDataset(img_path=img_path, gt_path=gt_img_path, txt_file=train_txt, transform=train_transform)
val_data = VOCSegDataset(img_path=img_path, gt_path=gt_img_path, txt_file=val_txt, transform=val_transform)

train_dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=32, shuffle=False)

for epoch in range(15):
    model.train()
    train_loss = 0
    loss_val = 0
    epoch_step = 0
    for images, targets in tqdm(train_dataloader, desc='training'):
        images = images.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        losses = criterion(outputs, targets)
        losses.backward()
        optimizer.step()
        train_loss += losses.item()
        epoch_step += 1

    print(f"tran losses: {train_loss / epoch_step:4f}")

    epoch_step = 0
    with torch.no_grad():
        model.eval()
        for images, targets in tqdm(val_dataloader, desc='validation'):
            images = images.cuda()
            outputs = model(images)
            loss = criterion(outputs['out'], targets)
            loss_val += loss.item()
            epoch_step += 1

        print(f"val losses: {loss_val / epoch_step:4f}")

    torch.save(model.state_dict(), f"./model/{epoch}_{loss_val / epoch_step:4f}.pth")
