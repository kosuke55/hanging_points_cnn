#!/usr/bin/env python
# coding: utf-8

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import skrobot
import numpy as np
import cv2
import cameramodels
import os
import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


def draw_axis(img, R, t, K):
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32(
        [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(
        img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()),
        (0, 0, 255), 1)
    img = cv2.line(
        img,
        tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()),
        (0, 255, 0), 1)
    img = cv2.line(
        img,
        tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()),
        (255, 0, 0), 1)
    return img


def find_contour_center(img):
    ret, thresh = cv2.threshold(img.copy(), int(255 * 0.5), 255, 0)
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(img, 1, 2)
    print(len(contours))
    cx_max = 0
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if cx_max < cx:
            cx_max = cx
            cx_result = cx
            cy_result = cy

    return cx_result, cy_result


data_path = '/media/kosuke/SANDISK/meshdata/Hanging-ObjectNet3D-DoubleFaces/rotations'
idx = 1
data_name_list = os.listdir(os.path.join(data_path, 'color'))
data_name_list.sort()

try:
    for data_name in data_name_list:
        color = cv2.imread(
            os.path.join(data_path, "color_raw/", data_name))

        heatmap = cv2.imread(
            os.path.join(data_path, "heatmap/", data_name),
            cv2.IMREAD_GRAYSCALE).astype(np.float32)

        rotations = np.load(
            os.path.join(data_path, "rotations/",
                         os.path.splitext(data_name)[0]) + ".npy")

        hanging_point_depth = np.load(
            os.path.join(data_path, "hanging_points_depth/",
                         os.path.splitext(data_name)[0]) + ".npy")

        clip_info = np.load(
            os.path.join(data_path, "clip_info/",
                         os.path.splitext(data_name)[0]) + ".npy")

        intrinsics = np.load(
            os.path.join(data_path, "intrinsics/intrinsics.npy"))

        xmin = clip_info[0]
        xmax = clip_info[1]
        ymin = clip_info[2]
        ymax = clip_info[3]

        heatmap_filtered = np.zeros_like(heatmap)

        confidence_thresh = np.max(heatmap) - 5
        heatmap_filtered[heatmap < confidence_thresh] = 0
        heatmap_filtered[heatmap >= confidence_thresh] = 255
        heatmap_filtered = heatmap_filtered.astype(np.uint8)

        scale_x = (xmax - xmin) / float(heatmap.shape[1])
        scale_y = (ymax - ymin) / float(heatmap.shape[0])

        cx, cy = find_contour_center(heatmap_filtered)
        rotation = rotations[cy, cx]
        depth = hanging_point_depth[cy, cx] * 0.001

        cameramodel = cameramodels.PinholeCameraModel.from_intrinsic_matrix(
            intrinsics, color.shape[0], color.shape[1])
        hanging_point_pose = np.array(
            cameramodel.project_pixel_to_3d_ray(
                [int(cx * scale_x + xmin),
                 int(cy * scale_y + ymin)])) * depth

        draw_axis(color,
                  skrobot.coordinates.math.quaternion2matrix(rotation),
                  hanging_point_pose,
                  intrinsics)

        cv2.circle(color, (int(cx * scale_x + xmin),
                           int(cy * scale_y + ymin)),
                   15, (255, 255, 255), thickness=1)
        cv2.rectangle(color, (xmin, ymin), (xmax, ymax), (255, 255, 0))

        # cv2.imshow('heatmap_filtered', heatmap_filtered)
        # cv2.imshow('heatmap_filtered_orig', heatmap_filtered_orig)
        # cv2.imshow('heatmap_filtered_orig_size', heatmap_filtered_orig_size)
        cv2.imshow('color', color)
        cv2.waitKey(5000)

        # cv2.imwrite('image.png', heatmap)
        cv2.imwrite('color.png', color)

except KeyboardInterrupt:
    pass
