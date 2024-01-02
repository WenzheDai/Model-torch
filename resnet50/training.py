
import cv2
import torch
import torchvision
import numpy as np
import glob
from torch import nn
from tqdm import tqdm
from torchvision import transforms

from resnet50 import Resnet50
from dataloader import MyDataLoader, MyDataset


class Train(object):
    def __init__(self, classes, train_data_loader, test_data_loader):
        self.model_obj = Resnet50(classes)

        self.net = self.model_obj.net.cuda()

        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=1e-3)

        self.loss_fun = nn.CrossEntropyLoss().cuda()

        self.train_data_loader = train_data_loader

        self.test_data_loader = test_data_loader

    def train_model(self, num_epochs):

        for epoch in range(num_epochs):
            self.net.train()
            for idx, (x, label) in tqdm(enumerate(self.train_data_loader)):
                x, label = x.cuda(), label.cuda()

                y = self.net(x)

                loss = self.loss_fun(y, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(epoch, f"loss:{loss.item()}")

            self.net.eval()
            with torch.no_grad():
                total_correct = 0
                total_num = 0
                for x, label in tqdm(self.test_data_loader):
                    x, label = x.cuda(), label.cuda()
                    y = self.net(x)

                    pred = y.argmax(dim=1)

                    correct = torch.eq(pred, label).float().sum().item()
                    total_correct += correct
                    total_num += x.size(0)

                acc = total_correct / total_num
                print(epoch, f"test acc: {acc}")
                torch.save(self.net.state_dict(), f"{round(acc, 3)}.pth")
    

def get_data():
    # 数据下载
    train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True)
    train_data_imgs = train_data.data
    train_data_idx_class = {v: k for k, v in train_data.class_to_idx.items()}
    train_data_labels = [train_data_idx_class[idx] for idx in train_data.targets]

    for idx, img in enumerate(train_data_imgs):
        cv2.imwrite(f"./train_data/image/{idx}.png", np.array(img))

    with open("./train_data/labels.txt", "w") as f:
        f.writelines(train_data_labels)

    test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True)
    test_data_imgs = test_data.data
    test_data_idx_class = {v: k for k, v in test_data.class_to_idx.items()}
    test_data_labels = [test_data_idx_class[idx] for idx in test_data.targets]

    for idx, img in enumerate(test_data_imgs):
        cv2.imwrite(f"./test_data/image/{idx}.png", np.array(img))

    with open("./test_data/labels.txt", "w") as f:
        f.writelines(test_data_labels)


if __name__ == '__main__':
    get_data()
    train_data_imgs_path = glob.glob("./train_data/image/*")
    test_data_imgs_path = glob.glob("./test_data/image/*")

    with open("./train_data/labels.txt", "rb") as f:
        train_labels = f.readline()
    
    with open("./test_data/labels.txt") as f:
        test_labels = f.readline()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, 224)
    ])
    
    train_dataset = MyDataset(train_data_imgs_path, train_labels, transform)
    test_dataset = MyDataset(test_data_imgs_path, test_labels)

    my_data_loader = MyDataLoader(train_dataset, test_dataset)

    train_data_loader, test_data_loader = my_data_loader.setup(batch_size=64, shuffle=True)

    train = Train(10, train_data_loader, test_data_loader)

    train.train_model(10)
