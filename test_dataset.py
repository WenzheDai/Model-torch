
from typing import Any
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

import cv2
import numpy as np


transform = transforms.Compose([
    transforms.ToTensor()
])


def default_loader(path):
    img_pil = Image.open(path)
    img_tensor = transform(img_pil)
    return img_tensor


class MyDataset(Dataset):
    def __init__(self, data_path, labels) -> None:
        self.data_tensors = [default_loader(each_data_path) for each_data_path in data_path]
        # self.data_tensors = data_tensors
        self.labels = labels

    def __getitem__(self, index) -> Any:
        img = self.data_tensors[index]
        label = self.labels[index]

        return img, label

    def __len__(self):
        return len(self.data_tensors)



def get_data():
    # train_data = torchvision.datasets.MNIST("./data", train=True, download=True)
    test_data = torchvision.datasets.MNIST("./data", train=False, download=True)

    data = test_data.data

    for idx, img in enumerate(data):
        cv2.imwrite(f"./image/{idx}.png", np.array(img))

    labels = test_data.targets
    labels = [str(label) for label in labels]

    with open("lables.txt", "w") as f:
        f.writelines(labels)

    return data, labels


# data, labels = get_data()

data_path = []
import glob
data_path = glob.glob("./image/*")
with open("./lables.txt", "r") as f:
    labels = f.readline()

print(len(data_path))
print(len(labels))

my_dataset = MyDataset(data_path, labels)

data_loader = DataLoader(my_dataset, batch_size=16, shuffle=True)

for data, label in data_loader:
    print(data)
    print(label)
