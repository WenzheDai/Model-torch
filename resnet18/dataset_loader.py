

import os
from PIL import Image
from torch.utils.data import Dataset


class DogCatDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.label_name = {"cat": 0, "dog": 1}
        self.root_path = root_path
        self.transform = transform
        self.train_img_name = None
        self.train_img_label = None
        self.get_train_img_info()

    def __getitem__(self, idx):
        self.img = Image.open(os.path.join(self.root_path, self.train_img_name[idx]).replace("\\","/"))

        if self.transform is not None:
            self.img = self.transform(self.img)
        self.label = self.train_img_label[idx]

        return self.img, self.label

    def __len__(self):
        return len(self.train_img_name)

    def get_train_img_info(self):
        self.train_img_name = os.listdir(self.root_path)
        self.train_img_label = [0 if 'cat' in img_name else 1 for img_name in self.train_img_name]
