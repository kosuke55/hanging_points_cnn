#!/usr/bin/env python
# coding: utf-8

import os
import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


def load_dataset(data_path, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor()])
    hp_data = HangingPointsDataset(data_path, transform)

    train_size = int(0.9 * len(hp_data))
    test_size = len(hp_data) - train_size
    # train_size = 1
    # test_size = 2
    train_dataset, test_dataset = random_split(hp_data, [train_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_dataloader, test_dataloader


def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size) * n + data.ravel()
    buf.ravel()[nmsk] = 1
    return buf


class HangingPointsDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(os.listdir(os.path.join(self.data_path, 'color')))

    def __getitem__(self, idx):
        data_name = os.listdir(os.path.join(self.data_path, 'color'))[idx]
        # depth = cv2.imread(os.path.join(self.data_path, "depth_bgr/", data_name)).astype(np.float32)
        depth = np.load(
            os.path.join(self.data_path, "depth/",
                         os.path.splitext(data_name)[0]) + ".npy").astype(np.float32) * 0.001

        clip_info = np.load(
            os.path.join(self.data_path, "clip_info/",
                         os.path.splitext(data_name)[0]) + ".npy")
        in_feature = depth

        confidence = cv2.imread(
            os.path.join(self.data_path, "heatmap/", data_name),
            cv2.IMREAD_GRAYSCALE).astype(np.float32)
        # print('confidence max', np.max(confidence))
        confidence /= 255.
        # print('confidence max 2', np.max(confidence))

        hanging_point_depth = np.load(
            os.path.join(self.data_path, "hanging_points_depth/",
                         os.path.splitext(data_name)[0]) + ".npy").astype(np.float32) * 0.001

        rotations = np.load(
            os.path.join(self.data_path, "rotations/",
                         os.path.splitext(data_name)[0]) + ".npy").astype(np.float32)

        # import ipdb; ipdb.set_trace()

        # print("confidence ", confidence.shape,
        #       "rotations ", rotations.shape,
        #       "hp depth", hanging_point_depth.shape)

        ground_truth = np.concatenate(
            [confidence[..., None],
             hanging_point_depth[..., None],
             rotations], axis=2)

        # print("ground_truth ", ground_truth.shape)

        if self.transform:
            in_feature = self.transform(in_feature)
            ground_truth = self.transform(ground_truth)

        return in_feature, clip_info, ground_truth
