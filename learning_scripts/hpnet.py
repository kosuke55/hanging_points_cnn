#!/usr/bin/env python3
# coding: utf-8

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import math
import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import RoIAlign

from resnet import resnet18
from utils.rois_tools import find_rois


class Conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, bias=True):
        super(Conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        output = self.relu(x)
        return output


class Decoder(nn.Module):

    def __init__(self, n_class=1):
        super().__init__()
        self.n_class = 1

        self.conv1 = Conv2DBatchNormRelu(
            512, 128, kernel_size=3, stride=1, padding=1)

        self.deconv2 = nn.ConvTranspose2d(
            128, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = Conv2DBatchNormRelu(
            128, 128, kernel_size=3, stride=1, padding=1)

        self.deconv3 = nn.ConvTranspose2d(
            128, self.n_class, kernel_size=16, stride=8, padding=4)
        self.conv3 = Conv2DBatchNormRelu(
            self.n_class, self.n_class, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.conv2(self.deconv2(h))
        h = self.conv3(self.deconv3(h))
        return h


class HPNET(nn.Module):
    def __init__(self, config=None):
        nn.Module.__init__(self)

        if config is None:
            config = {
                'feature_compress': 1 / 16,
                'num_class': 6,
                'pool_out_size': 8,
            }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_compress = config['feature_compress']
        self.pool_out_size = config['pool_out_size']
        self.n_class = config['num_class']

        resnet = resnet18()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        self.decoder = Decoder(self.n_class)

        self.conv_to_head = nn.Sequential(
            Conv2DBatchNormRelu(
                512, 128, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128 * 8 * 8),
            nn.Linear(128 * 8 * 8, 5)
        )
        self.roi_align = RoIAlign(8, 1 / 32, -1)

    def forward(self, x):
        h = x
        feature = self.feature_extractor(h)

        confidence = self.decoder(feature)

        self.rois_list = find_rois(confidence)
        if self.rois_list is None:

            self.rois_list = [torch.tensor(
                [[0, 0, 0, 0]], dtype=torch.float32).to(
                    self.device) for _ in range(confidence.shape[0])]
        rois = self.roi_align(feature, self.rois_list)
        depth_and_rotation = self.conv_to_head(rois)

        return confidence, depth_and_rotation
