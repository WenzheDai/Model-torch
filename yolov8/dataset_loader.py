

import os
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, data_img, data_label):
        self.images = os.listdir(data_img)
        self.labels = os.listdir(data_label)

    def __getitem__(self, idx):
        pass
