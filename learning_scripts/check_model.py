#!/usr/bin/env python3
# coding: utf-8

from hpnet import HPNET
from torchvision import transforms
from torchsummary import summary
import torch
import numpy as np
import cv2
import os
import sys

# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

from utils.rois_tools import expand_roi, find_rois


config = {
    'feature_stride': 16,
    'feature_compress': 1 / 16,
    'num_feature_channel': 256,
    'num_fc7_channel': 512,
    'num_rpn_channel': 512,
    'num_anchor': 9,
    'score_top_n': 100,
    'nms_top_n': 50,
    'nms_thresh': 0.7,
    'pool_out_size': 8,
    'num_class': 1,
    'radios': (0.5, 1, 2),
    'scales': (4, 8, 16),
}

data_path = '/media/kosuke/SANDISK/meshdata/Hanging-ObjectNet3D-DoubleFaces/rotations_0514_1000'
pretrained_model = '/media/kosuke/SANDISK/hanging_points_net/checkpoints/resnet/hpnet_bestmodel_20200517_0426.pt'

for idx in range(1):
    data_name = sorted(os.listdir(os.path.join(data_path, 'color')))[idx]

    # data_name = data_name.sort()

    print(data_name)

    # data_name = data_name.sort()[0]

    depth = np.load(
        os.path.join(data_path, "depth/",
                     os.path.splitext(data_name)[0]) + ".npy").astype(np.float32) * 0.001
    color = cv2.imread(os.path.join(data_path, "color/", data_name)).astype(np.float32)

    confidence_gt_img = cv2.imread(
        os.path.join(data_path, "heatmap/", data_name),
        cv2.IMREAD_GRAYSCALE).astype(np.float32)
    confidence_gt = confidence_gt_img / 255.

    model = HPNET(config).cuda()
    model.load_state_dict(torch.load(pretrained_model), strict=False)
    # image = torch.rand((4, 1, 256, 256)).cuda()
    transform = transforms.Compose([
        transforms.ToTensor()])
    depth = transform(depth)[None, ...].cuda()
    # print(depth.shape)
    confidence, depth_and_rotation = model.forward(depth)

    confidence_gt = transform(confidence_gt)[None, ...].cuda()
    gt_rois_list = find_rois(confidence_gt)
    print('gt_rois', gt_rois_list)
    print('rois', model.rois_list)

    print('confidence.shape', confidence.shape)
    print('depth_and_rotation.shape', depth_and_rotation.shape)

    print(model.rois_list)
    confidence = confidence[0, 0, ...]
    confidence = confidence.cpu().detach().numpy().copy() * 255
    confidence = confidence.astype(np.uint8)

    confidence_bgr = cv2.cvtColor(confidence, cv2.COLOR_GRAY2BGR)

    for rois in model.rois_list:
        # print('--')
        for roi in rois:
            # print(roi)
            confidence_bgr = cv2.rectangle(
                confidence_bgr, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 3)
            color = cv2.rectangle(
                color, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 3)

    cv2.imwrite('debug/confidence{:03}.png'.format(idx), confidence_bgr)
    cv2.imwrite('debug/color{:03}.png'.format(idx), color)
    # cv2.imwrite('confidence.png', confidence)

    # summary(model, (1, 256, 256)) # summary(model,(channels,H,W))
