#!/usr/bin/env python3
# coding: utf-8

# from __future__ import absolute_import
# from __future__ import division

import sys
import os.path as osp

import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import RoIAlign

try:
    import cv2
except ImportError:
    for path in sys.path:
        if '/opt/ros/' in path:
            print('sys.path.remove({})'.format(path))
            sys.path.remove(path)
            import cv2
            sys.path.append(path)

from learning_scripts.resnet import resnet18
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

    def __init__(self, output_channels=1, feature_extractor_name='resnet50'):
        super(Decoder, self).__init__()
        self.output_channels = output_channels
        if feature_extractor_name == 'resnet50':
            self.feature_extractor_out_channels = 2048
        elif feature_extractor_name == 'resnet18':
            self.feature_extractor_out_channels = 512

        self.conv1 = Conv2DBatchNormRelu(
            self.feature_extractor_out_channels, 128, kernel_size=3, stride=1, padding=1)

        self.deconv2 = nn.ConvTranspose2d(
            128, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = Conv2DBatchNormRelu(
            128, 128, kernel_size=3, stride=1, padding=1)

        if feature_extractor_name == 'resnet50':
            self.deconv3 = nn.ConvTranspose2d(
                128, self.output_channels, kernel_size=32, stride=16, padding=8)  # *16
        elif feature_extractor_name == 'resnet18':
            self.deconv3 = nn.ConvTranspose2d(
                128, self.output_channels, kernel_size=16, stride=8, padding=4)  # *8
        self.conv3 = Conv2DBatchNormRelu(
            self.output_channels, self.output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = x
        # h = self.conv0(h)
        h = self.conv1(h)
        h = self.conv2(self.deconv2(h))
        h = self.conv3(self.deconv3(h))
        return h


class HPNET(nn.Module):
    def __init__(self, config=None):
        nn.Module.__init__(self)

        if config is None:
            config = {
                'output_channels': 1,
                'feature_extractor_name': 'resnet50',
                'confidence_thresh': 0.3,
                'use_bgr': True,
                'use_bgr2gray': True,
            }

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.output_channels = config['output_channels']
        self.confidence_thresh = config['confidence_thresh']

        if config['feature_extractor_name'] == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.feature_extractor_out_channels = 2048
            self.feature_extractor_out_size = 8
            self.roi_align_spatial_scale = 1 / 32.
        elif feature_extractor_name == 'resnet18':
            resnet = resnet18()
            self.feature_extractor_out_channels = 512
            self.feature_extractor_out_size = 8
            self.roi_align_spatial_scale = 1 / 16.

        if config['use_bgr']:
            if config['use_bgr2gray']:
                in_channnels = 2
            else:
                in_channnels = 6
        else:
            in_channnels = 1

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channnels, 64, kernel_size=7, stride=2, padding=3),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        self.decoder = Decoder(self.output_channels)

        self.conv_to_head = nn.Sequential(
            Conv2DBatchNormRelu(
                self.feature_extractor_out_channels, 128,
                kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128 * 8 * 8),
            nn.Linear(128 * 8 * 8, 5)
        )

        self.roi_align = RoIAlign(
            self.feature_extractor_out_size, self.roi_align_spatial_scale, -1)

    def forward(self, x):
        h = x
        feature = self.feature_extractor(h)
        confidence = self.decoder(feature)

        self.rois_list, self.rois_center_list = find_rois(
            confidence, confidence_thresh=self.confidence_thresh)

        if self.rois_list is None:
            self.rois_list = [torch.tensor(
                [[0, 0, 0, 0]], dtype=torch.float32).to(
                    self.device) for _ in range(confidence.shape[0])]
        rois = self.roi_align(feature, self.rois_list)
        depth_and_rotation = self.conv_to_head(rois)

        return confidence, depth_and_rotation
