
import torch
import torchvision
from torch import nn
from tqdm import tqdm

from resnet50 import Resnet50
from dataloader import MyDataLoader


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


if __name__ == '__main__':
    # 数据下载
    train_data = torchvision.datasets.CIFAR10("./data",
                                              transform=torchvision.transforms.Compose(
                                                  [torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Resize(64)]),
                                              train=True, download=True)

    test_data = torchvision.datasets.CIFAR10("./data",
                                             transform=torchvision.transforms.Compose(
                                                 [torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Resize(64)]),
                                             train=False, download=True)

    my_data_loader = MyDataLoader(train_data, test_data)

    train_data_loader, test_data_loader = my_data_loader.setup(batch_size=64, shuffle=True)

    train = Train(10, train_data_loader, test_data_loader)

    train.train_model(10)
