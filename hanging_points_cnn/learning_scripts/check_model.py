#!/usr/bin/env python3
# coding: utf-8

import os
import os.path as osp
import sys

import cameramodels
import numpy as np
from skrobot.coordinates.math import quaternion2matrix
import torch
from torchsummary import summary
from torchvision import transforms

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))  # noqa:
from hpnet import HPNET
from utils.visualize import colorize_depth, draw_axis

for path in sys.path:
    if 'opt/ros/' in path:
        print('sys.path.remove({})'.format(path))
        sys.path.remove(path)
        import cv2
        sys.path.append(path)
    else:
        import cv2

config = {
    'feature_compress': 1 / 16,
    'num_class': 1,
    'pool_out_size': 8,
    'confidence_thresh': 0.3,
}

data_path = '/media/kosuke/SANDISK/meshdata/ycb_hanging_object/per500/'
pretrained_model = '/media/kosuke/SANDISK/hanging_points_net/checkpoints/resnet/hpnet_latestmodel_20200606_0444.pt'

intrinsics = np.load(
    osp.join(data_path, "intrinsics/intrinsics.npy"))
cameramodel = cameramodels.PinholeCameraModel.from_intrinsic_matrix(
    intrinsics, 1080, 1920)
model = HPNET(config).cuda()
model.load_state_dict(torch.load(pretrained_model), strict=False)

for idx in range(10000):

    # idx = 4497
    data_name = sorted(os.listdir(
        os.path.join(data_path, 'heatmap')))[idx]

    depth = np.load(
        os.path.join(data_path, "depth/",
                     os.path.splitext(data_name)[0]) + ".npy").astype(np.float32) * 0.001
    clip_info = np.load(
        os.path.join(data_path, "clip_info/",
                     os.path.splitext(data_name)[0]) + ".npy")
    xmin = clip_info[0]
    xmax = clip_info[1]
    ymin = clip_info[2]
    ymax = clip_info[3]

    confidence_gt = cv2.imread(
        os.path.join(data_path, "heatmap/", data_name),
        cv2.IMREAD_GRAYSCALE).astype(np.float32)
    # confidence_gt /= 255.
    # # image = torch.rand((4, 1, 256, 256)).cuda()
    transform = transforms.Compose([
        transforms.ToTensor()])
    depth = transform(depth)[None, ...].cuda()
    # # print(depth.shape)
    confidence, depth_and_rotation = model.forward(depth)

    if len(model.rois_list[0]) < 2:
        continue

    # confidence_gt = transform(confidence_gt)[None, ...].cuda()
    # gt_rois_list = find_rois(confidence_gt)
    # print('gt_rois', gt_rois_list)
    # print('rois', model.rois_list)

    # print('confidence.shape', confidence.shape)
    # print('depth_and_rotation.shape', depth_and_rotation.shape)

    # print(model.rois_list)
    confidence = confidence[0, 0, ...]
    confidence = confidence.cpu().detach().numpy().copy() * 255
    confidence = confidence.astype(np.uint8)

    confidence_bgr = cv2.cvtColor(confidence, cv2.COLOR_GRAY2BGR)
    print(depth.shape)
    depth = depth.cpu().detach().numpy().copy()[0, 0, ...] * 1000

    depth_bgr = colorize_depth(depth.copy(), 100, 1500)
    axis_pred = depth_bgr.copy()
    axis_large_pred = np.zeros((1080, 1920, 3))
    axis_large_pred[ymin:ymax, xmin:xmax]\
        = cv2.resize(axis_pred, (xmax - xmin, ymax - ymin))

    for i, roi in enumerate(model.rois_list[0]):
        if roi.tolist() == [0, 0, 0, 0]:
            continue
        roi = roi.cpu().detach().numpy().copy()
        cx = int((roi[0] + roi[2]) / 2)
        cy = int((roi[1] + roi[3]) / 2)
        dep = depth[int(roi[1]):int(roi[3]),
                    int(roi[0]):int(roi[2])]
        dep = np.median(dep[np.where(
            np.logical_and(dep > 200, dep < 1000))]).astype(np.uint8)
        confidence_bgr = cv2.rectangle(
            confidence_bgr, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 3)

        q = depth_and_rotation[i, 1:].cpu().detach().numpy().copy()
        q /= np.linalg.norm(q)
        pixel_point = [int(cx * (xmax - xmin) / float(256) + xmin),
                       int(cy * (ymax - ymin) / float(256) + ymin)]
        hanging_point_pose = np.array(
            cameramodel.project_pixel_to_3d_ray(
                pixel_point)) * float(dep * 0.001)

        draw_axis(axis_large_pred,
                  quaternion2matrix(q),
                  hanging_point_pose,
                  intrinsics)
        axis_pred = cv2.resize(axis_large_pred[ymin:ymax, xmin:xmax],
                               (256, 256)).astype(np.uint8)

        # print('--')
        # for roi in rois:
        #     print(roi)
        #     confidence_bgr = cv2.rectangle(
        #         confidence_bgr, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 3)
        #     color = cv2.rectangle(
        #         color, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 3)

    cv2.imwrite('debug/confidence{:03}.png'.format(idx), confidence_bgr)
    cv2.imwrite('debug/confidence_gt{:03}.png'.format(idx), confidence_gt)
    cv2.imwrite('debug/axis_pred{:03}.png'.format(idx), axis_pred)
    # cv2.imwrite('debug/color{:03}.png'.format(idx), color)
    # # cv2.imwrite('confidence.png', confidence)

    # summary(model, (1, 256, 256))  # summary(model,(channels,H,W))
