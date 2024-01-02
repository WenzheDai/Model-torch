
from typing import Any, List, Callable
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils import data
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, data_path, labels, transform: Callable) -> None:
        self.data_tensors = [self.default_loader(each_data_path) for each_data_path in data_path]
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index) -> Any:
        img = self.data_tensors[index]
        label = self.labels[index]

        return img, label

    def __len__(self):
        return len(self.data_tensors)

    def default_loader(self, path):
        img_pil = Image.open(path)
        img_tensor = self.transform(img_pil)
        return img_tensor


class MyDataLoader(object):
    def __init__(self, train, test):
        self.train_data = train
        self.test_data = test

    def setup(self, batch_size, shuffle=False):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        train_data_loder = DataLoader(
            # dataset=transform(self.train_data),
            dataset=self.train_data,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        test_data_loader = DataLoader(
            # dataset=transform(self.test_data),
            dataset=self.test_data,
            batch_size=batch_size,
            shuffle=shuffle
        )

        return train_data_loder, test_data_loader
