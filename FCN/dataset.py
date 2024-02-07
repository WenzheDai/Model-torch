
import os

import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image


class VOCSegDataset(Dataset):
    def __init__(self, img_path, gt_path, txt_file, transform=None):
        super().__init__()
        self.transform = transform

        with open(txt_file, 'r') as f:
            data = [data.strip() for data in f.readlines() if len(data.strip()) > 0]

        self.img_files = [os.path.join(img_path, i + '.jpg') for i in data]
        self.gt_files = [os.path.join(gt_path, i + '.png') for i in data]

    def __getitem__(self, idx):
        # img = Image.open(self.img_files[idx])
        # target = Image.open(self.gt_files[idx])
        img = cv2.imread(self.img_files[idx])
        target = cv2.imread(self.gt_files[idx], cv2.IMREAD_GRAYSCALE)

        transformed = self.transform(image=img, mask=target)
        img = transformed['image']
        target = transformed['mask'].to(torch.long)

        return img, target

    def __len__(self):
        return len(self.img_files)
