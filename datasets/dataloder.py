#!usr/bin/python3
# -*- coding: utf-8 -*- 
"""
Project: resnet_zml
File: dataloder.py
IDE: PyCharm
Creator: morei
Email: zhangmengleiba361@163.com
Create time: 2021-02-22 16:38
Introduction:
"""

import os
import pandas as pd
from PIL import Image as pil_image
from torch.utils.data import Dataset


class LoadDataset(Dataset):

    def __init__(self, csv_path, img_transform=None, label_transform=None):
        self.csv_path = csv_path
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.img_df = pd.read_csv(self.csv_path)
        self.img_df.columns = ['relative_path', 'true_label']

    def __getitem__(self, index):
        img_path, label = self.img_df.iloc[index]
        img = pil_image.open(img_path)
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, label

    def __len__(self):
        return len(self.img_df)



