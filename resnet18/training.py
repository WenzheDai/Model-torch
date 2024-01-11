

import torch
from torch import nn
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset_loader import DogCatDataset
from model import Resnet18


def main():
    transformer = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor()
    ])
    tran_data = DogCatDataset(root_path=r"./dataset/CatDog/train", transform=transformer)
    train_dataloader = DataLoader(dataset=tran_data, batch_size=64, shuffle=True)

    model = Resnet18(num_classes=2)
    pretrained_weight = torch.hub.load_state_dict_from_url(
        url='https://download.pytorch.org/models/resnet18-5c106cde.pth', progress=True)

    del pretrained_weight['fc.weight']
    del pretrained_weight['fc.bias']

    model.load_state_dict(pretrained_weight, strict=False)

    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    LR = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.train()
    for epoch in range(20):
        loss_log = 0
        total_sample = 0
        train_correct_sample = 0
        for img, label in tqdm(train_dataloader):
            img, label = img.cuda(), label.cuda()

            outputs = model(img)
            optimizer.zero_grad()
            loss = criterion(outputs, label)
            loss.backend()

            optimizer.step()

            _, pred = torch.max(outputs, 1)
            total_sample += label.size(0)
            train_correct_sample += (label == pred).cpu().sum().numpy()

            loss_log += loss.item()

        accuracy = train_correct_sample / total_sample
        print(f"{epoch=}")
        print(f"{accuracy=}")
        print(f"loss: {loss_log / total_sample}")

        torch.save(model.state_dict(), f"./model/{epoch}-{accuracy}")

        scheduler.step()


if __name__ == '__main__':
    main()


