#!/usr/bin/env python
# coding: utf-8

import os
import sys
from pathlib import Path

import imgaug.augmenters as iaa
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from hanging_points_cnn.utils.image import colorize_depth, normalize_depth

for path in sys.path:
    if 'opt/ros/' in path:
        print('sys.path.remove({})'.format(path))
        sys.path.remove(path)
        import cv2
        sys.path.append(path)
    else:
        import cv2


def load_dataset(data_path, batch_size, use_bgr, use_bgr2gray, depth_range):
    transform = transforms.Compose([
        transforms.ToTensor()])
    hp_data = HangingPointsDataset(
        data_path, transform, use_bgr, use_bgr2gray, depth_range)

    train_size = int(0.9 * len(hp_data))
    test_size = len(hp_data) - train_size
    # train_size = 1
    # test_size = 1
    train_dataset, test_dataset = random_split(
        hp_data, [train_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_dataloader, test_dataloader


class HangingPointsDataset(Dataset):
    def __init__(self, data_path, transform=None,
                 use_bgr=True, use_bgr2gray=True, depth_range=[0.2, 0.7], test=False):

        self.data_path = data_path
        self.transform = transform
        self.file_paths = list(
            sorted(Path(self.data_path).glob("*/depth/*.npy")))
        self.use_bgr = use_bgr
        if use_bgr2gray:
            self.use_bgr = True
        self.use_bgr2gray = use_bgr2gray
        self.depth_range = depth_range
        self.test = test

        self.aug_seq = iaa.Sequential([
            iaa.Dropout([0, 0.3]),
            iaa.GaussianBlur((0, 1.0)),
        ], random_order=True)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        depth_filepath = self.file_paths[idx]

        depth = np.load(depth_filepath).astype(np.float32)
        depth = self.aug_seq.augment_image(depth)

        # r = np.random.randint(20)
        # kernel = np.ones((r, r), np.uint8)
        # depth = cv2.dilate(depth, kernel, iterations=1)

        # r = np.random.randint(20)
        # r = r if np.mod(r, 2) else r + 1
        # depth = cv2.GaussianBlur(depth, (r, r), 10)

        if self.use_bgr:
            depth_bgr = colorize_depth(depth.copy(), 100, 1500)
            color = cv2.imread(
                str(depth_filepath.parent.parent / 'color' /
                    depth_filepath.with_suffix('.png').name),
                cv2.IMREAD_COLOR)
            color = self.aug_seq.augment_image(color)
            if self.use_bgr2gray:
                # 0~1
                gray = cv2.cvtColor(
                    color, cv2.COLOR_BGR2GRAY)[..., None] / 255.
                gray = gray.astype(np.float32)
                normalized_depth = normalize_depth(
                    depth, self.depth_range[0], self.depth_range[1])[..., None]

                # -1~1
                # gray = gray * 2 - 1
                # depth = depth * 2 - 1

                # 2 channels
                in_feature = np.concatenate((normalized_depth, gray), axis=2)
            else:
                # 6 channels
                in_feature = np.concatenate((depth_bgr, color), axis=2)
        else:
            in_feature = depth * 0.001

        if self.test:
            if self.transform:
                in_feature = self.transform(in_feature)

        # clip_info = np.load(
        #     depth_filepath.parent.parent / 'clip_info' / depth_filepath.name)
        camera_info_path = str(
            depth_filepath.parent.parent /
            'camera_info' / depth_filepath.with_suffix('.yaml').name)

        confidence = cv2.imread(
            str(depth_filepath.parent.parent / 'heatmap' /
                depth_filepath.with_suffix('.png').name),
            cv2.IMREAD_GRAYSCALE).astype(np.float32)
        confidence /= 255.

        hanging_point_depth = np.load(
            depth_filepath.parent.parent / 'hanging_points_depth' /
            depth_filepath.name).astype(np.float32)

        hanging_point_depth = normalize_depth(
            hanging_point_depth, self.depth_range[0], self.depth_range[1])

        rotations = np.load(
            depth_filepath.parent.parent / 'rotations' /
            depth_filepath.name).astype(np.float32)

        ground_truth = np.concatenate(
            [confidence[..., None],
             hanging_point_depth[..., None],
             rotations], axis=2)

        if self.transform:
            in_feature = self.transform(in_feature)
            depth = self.transform(depth)
            ground_truth = self.transform(ground_truth)

        return in_feature, depth, camera_info_path, ground_truth
