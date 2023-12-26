
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


class MyDataset(object):
    pass


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
