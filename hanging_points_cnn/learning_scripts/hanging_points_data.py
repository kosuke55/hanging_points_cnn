#!/usr/bin/env python
# coding: utf-8

import os
import sys
from pathlib import Path
from PIL import Image

import numpy as np
import imgaug.augmenters as iaa
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split

from hanging_points_cnn.utils.image import colorize_depth
from hanging_points_cnn.utils.image import normalize_depth
from hanging_points_cnn.utils.image import resize_np_img
from hanging_points_cnn.utils.random_eraser import get_random_eraser


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
    val_size = len(hp_data) - train_size

    train_dataset, val_dataset = random_split(
        hp_data, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_dataloader, val_dataloader


def load_test_dataset(data_path, use_bgr, use_bgr2gray, depth_range):
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = HangingPointsDataset(
        data_path, transform, use_bgr, use_bgr2gray, depth_range, test=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1)

    return test_dataloader


class HangingPointsDataset(Dataset):
    def __init__(self, data_path, transform=None,
                 use_bgr=True, use_bgr2gray=True,
                 depth_range=[0.2, 0.7], test=False):
        self.test = test
        self.data_path = data_path
        self.transform = transform
        if self.test:
            self.file_paths = list(
                sorted(Path(self.data_path).glob("depth/*.npy")))
        else:
            # data_path/class/fancy/depth/*.npy
            self.file_paths = list(
                sorted(Path(self.data_path).glob("*/*/depth/*.npy")))
        self.use_bgr = use_bgr
        if use_bgr2gray:
            self.use_bgr = True
        self.use_bgr2gray = use_bgr2gray
        self.depth_range = depth_range
        self.inshape = (256, 256)

        self.aug_seq = iaa.Sequential([
            iaa.Dropout([0, 0.2]),
            # iaa.GaussianBlur((0, 1.0)),
        ], random_order=True)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        depth_filepath = self.file_paths[idx]
        depth = np.load(depth_filepath).astype(np.float32)

        if self.test:
            depth = resize_np_img(depth, self.inshape, Image.NEAREST)
        else:
            depth = self.aug_seq.augment_image(depth)
            nonzero_depth = depth.copy()
            nonzero_depth[nonzero_depth == 0] =depth.max()
            depth_eraser = get_random_eraser(
                p=0.9, v_l=nonzero_depth.min(), v_h=depth.max())
            depth = depth_eraser(depth)

        # r = np.random.randint(20)
        # kernel = np.ones((r, r), np.uint8)
        # depth = cv2.dilate(depth, kernel, iterations=1)

        # r = np.random.randint(20)
        # r = r if np.mod(r, 2) else r + 1
        # depth = cv2.GaussianBlur(depth, (r, r), 10)

        camera_info_path = str(
            depth_filepath.parent.parent /
            'camera_info' / depth_filepath.with_suffix('.yaml').name)

        if self.use_bgr:
            depth_bgr = colorize_depth(
                depth.copy(),
                self.depth_range[0], self.depth_range[1])
            color = cv2.imread(
                str(depth_filepath.parent.parent / 'color' /
                    depth_filepath.with_suffix('.png').name),
                cv2.IMREAD_COLOR)
            if self.test:
                color = resize_np_img(color, self.inshape)
            else:
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
                depth = self.transform(depth)
                return in_feature, depth, camera_info_path, 'dummy'

        # clip_info = np.load(
        #     depth_filepath.parent.parent / 'clip_info' / depth_filepath.name)

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
