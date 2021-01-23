#!/usr/bin/env python
# coding: utf-8

import os
import sys
from pathlib import Path
from PIL import Image

import numpy as np
import imgaug.augmenters as iaa
import torch
from hanging_points_generator.generator_utils import load_json
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split

from hanging_points_cnn.create_dataset.renderer import DepthMap
from hanging_points_cnn.create_dataset.renderer import RotationMap
from hanging_points_cnn.utils.image import colorize_depth
from hanging_points_cnn.utils.image import create_gradient_circle
from hanging_points_cnn.utils.image import depth_edges_erase
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


def collate_fn(batch):
    in_features = []
    depths = []
    camera_info_paths = []
    ground_truths = []
    annotations = []
    for in_feature, depth, camera_info_path, ground_truth, annotation in batch:
        in_features.append(in_feature)
        depths.append(depth)
        camera_info_paths.append(camera_info_path)
        ground_truths.append(ground_truth)
        annotations.append(annotation)

    in_features = torch.stack(in_features, dim=0)
    depths = torch.stack(depths, dim=0)
    # camera_info_paths = torch.stack(camera_info_paths, dim=0)
    ground_truths = torch.stack(ground_truths, dim=0)

    return in_features, depths, camera_info_paths, ground_truths, annotations


def load_dataset(data_path, batch_size, use_bgr, use_bgr2gray,
                 depth_range, object_list=None, data_augmentation=True):
    transform = transforms.Compose([
        transforms.ToTensor()])
    hp_data = HangingPointsDataset(
        data_path, transform, use_bgr, use_bgr2gray,
        depth_range, object_list=object_list,
        data_augmentation=data_augmentation)

    print('Load {} data'.format(len(hp_data)))
    train_size = int(0.9 * len(hp_data))
    val_size = len(hp_data) - train_size

    train_dataset, val_dataset = random_split(
        hp_data, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=False, num_workers=1, collate_fn=collate_fn)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=1, collate_fn=collate_fn)

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
                 depth_range=[0.2, 0.7], test=False,
                 object_list=None, data_augmentation=True):
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
            if object_list is not None:
                print('len(object_list)', len(object_list))
                self.file_paths = list(filter(
                    lambda path: any(target_object in str(path)
                                     for target_object in object_list),
                    self.file_paths))
                print('Select {} data in object_list'.format(
                    len(self.file_paths)))

        self.use_bgr = use_bgr
        if use_bgr2gray:
            self.use_bgr = True
        self.use_bgr2gray = use_bgr2gray
        self.depth_range = depth_range
        self.data_shape = (256, 256)
        self.data_augmentation = data_augmentation

        self.aug_seq = iaa.Sequential([
            iaa.Dropout([0, 0.8]),
            iaa.MultiplyElementwise((0.99, 1.01)),
            # iaa.GaussianBlur((0, 1.0)),
        ], random_order=True)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        depth_filepath = self.file_paths[idx]
        depth = np.load(depth_filepath).astype(np.float32)

        if self.test:
            depth = resize_np_img(depth, self.data_shape, Image.NEAREST)
        else:
            depth = depth_edges_erase(depth)
            depth = self.aug_seq.augment_image(depth)
            if self.data_augmentation:
                nonzero_depth = depth.copy()
                nonzero_depth[nonzero_depth == 0] = depth.max()
                depth_eraser = get_random_eraser(
                    p=0.9, s_l=0.1, s_h=0.5,
                    v_l=nonzero_depth.min(),
                    v_h=depth.max(),
                    pixel_level=True)
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
                color = resize_np_img(color, self.data_shape)
            # else:
            #     color = self.aug_seq.augment_image(color)
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
                return in_feature, depth, camera_info_path, 'dummy', 'dummy'

        annotation_data = load_json(
            str(depth_filepath.parent.parent / 'annotation' /
                depth_filepath.with_suffix('.json').name)
        )

        confidence = np.zeros(self.data_shape, dtype=np.uint32)
        # depth_map = DepthMap(
        #     self.data_shape[1],
        #     self.data_shape[0],
        #     circular=True)
        # rotation_map = RotationMap(self.data_shape[1], self.data_shape[0])

        for i, annotation in enumerate(annotation_data):
            px = annotation['xy'][0]
            py = annotation['xy'][1]
            annotation_data[i]['depth'] = normalize_depth(
                annotation_data[i]['depth'],
                self.depth_range[0], self.depth_range[1])
            # depth_value = annotation['depth']
            # quaternion = np.array(annotation['quaternion'])

            create_gradient_circle(
                confidence, py, px)
            # rotation_map.add_quaternion(
            #     px, py, quaternion)
            # depth_map.add_depth(px, py, depth_value)

        confidence = (confidence / confidence.max()).astype(np.float32)
        # rotations = rotation_map.rotations.astype(np.float32)
        # hanging_point_depth \
        #     = depth_map.on_depth_image(depth).astype(np.float32)

        # clip_info = np.load(
        #     depth_filepath.parent.parent / 'clip_info' / depth_filepath.name)

        # confidence = cv2.imread(
        #     str(depth_filepath.parent.parent / 'heatmap' /
        #         depth_filepath.with_suffix('.png').name),
        #     cv2.IMREAD_GRAYSCALE).astype(np.float32)
        # confidence /= 255.

        # hanging_point_depth = np.load(
        #     depth_filepath.parent.parent / 'hanging_points_depth' /
        #     depth_filepath.name).astype(np.float32)

        # hanging_point_depth = normalize_depth(
        #     hanging_point_depth, self.depth_range[0], self.depth_range[1])

        # rotations = np.load(
        #     depth_filepath.parent.parent / 'rotations' /
        #     depth_filepath.name).astype(np.float32)

        # ground_truth = np.concatenate(
        # [confidence[..., None],
        #  hanging_point_depth[..., None],
        #  rotations], axis=2)

        ground_truth = confidence[..., None]

        if self.transform:
            in_feature = self.transform(in_feature)
            depth = self.transform(depth)
            ground_truth = self.transform(ground_truth)

        return in_feature, depth, camera_info_path, ground_truth, annotation_data
