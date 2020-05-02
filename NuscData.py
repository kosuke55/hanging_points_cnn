#!/usr/bin/env python
# coding: utf-8

import os
import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


def load_dataset(data_path):
    transform = transforms.Compose([
        transforms.ToTensor()])
    nusc = NuscDataset(data_path, transform)

    train_size = int(0.9 * len(nusc))
    test_size = len(nusc) - train_size
    # train_size = 1
    # test_size = 2
    train_dataset, test_dataset = random_split(nusc, [train_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=1)
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=True, num_workers=1)

    return train_dataloader, test_dataloader


def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size) * n + data.ravel()
    buf.ravel()[nmsk] = 1
    return buf


class NuscDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(os.listdir(os.path.join(self.data_path, 'color')))

    def __getitem__(self, idx):
        data_name = os.listdir(os.path.join(self.data_path, 'color'))[idx]
        color = cv2.imread(os.path.join(self.data_path, "color/", data_name)).astype(np.float32)
        depth = cv2.imread(os.path.join(self.data_path, "depth/", data_name)).astype(np.float32)
        # color = cv2.resize(color, (640, 640))
        # depth = cv2.resize(depth, (640, 640))
        # print(color.shape)
        # print(depth.shape)
        in_feature = np.concatenate((color, depth), axis=2)

        ground_truth = cv2.imread(os.path.join(self.data_path, "annotation/", data_name), cv2.IMREAD_GRAYSCALE).astype(np.float32)

        if self.transform:
            in_feature = self.transform(in_feature)
            ground_truth = self.transform(ground_truth)

        return in_feature, ground_truth
