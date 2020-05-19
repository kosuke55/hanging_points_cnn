#!/usr/bin/env python3
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
# from layers.anchor_generation_layer import generate_anchors
# from layers.proposal_computation_layer import compute_proposals
# from layers.region_proposal_network import RegionProposalNetwor


def expand_roi(box, img_shape, scale=1.5):
    x, y, w, h = box
    wmax, hmax = img_shape
    xo = np.max([x - (scale - 1) * w / 2, 0])
    yo = np.max([y - (scale - 1) * h / 2, 0])
    wo = w * scale
    ho = h * scale
    if xo + wo >= wmax:
        wo = wmax - xo - 1
    if yo + ho >= hmax:
        ho = hmax - yo - 1

    return [xo, yo, wo, ho]


def find_rois(confidence,
              confidence_gt=None,
              confidence_thresh=0.5,
              area_thresh=1000,
              mode='train'):
    """Find rois

    gtとの比較はlossの方で行う.ここではconfidenceの推論からroiを提案すればよい.

    Parametersa
    ----------
    confidence : torch.tensor
        NCHW
    Returns
    -------
    rois : torch.tensor
        rois [[[x1, y1, x2, y2], ..., [x1, y1, x2, y2]],
              ... [x1, y1, x2, y2], ..., [x1, y1, x2, y2]]
        len(rois) = n (batch size)

        - example shape
        rois_list (2,)
        rois_list[0] torch.Size([46, 4])
        rois_list[1] torch.Size([38, 4])
        The first dimension of rois_list shows a batch,
        which contains (the number of roi, (dx, dy, dw, dh)).
    """

    # if mode == 'train':

    confidence = confidence.cpu().detach().numpy().copy()
    # confidence_gt = confidence_gt.detach().numpy().copy()

    rois = []
    for n in range(confidence.shape[0]):
        rois_n = None
        # confidence_mask = np.full_like(
        #     confidence_gt, 0.1, dtype=(np.float32))
        # confidence_mask[np.where(
        #     np.logical_and(
        #         confidence_gt[n, ...].transpose(1, 2, 0) > confidence_thresh,
        #         confidence_gt[n, ...].transpose(1, 2, 0) > confidence_thresh))] = 1.
        # confidence_mask = np.zeros((confidence.shape[2:]))
        # confidence_mask[np.where(
        #     confidence_gt[n, ...].transpose(1, 2, 0)) > confidence_thresh] = 1.
        confidence_mask = confidence[n, ...].transpose(1, 2, 0)
        confidence_mask[confidence_mask > 1] = 1
        confidence_mask[confidence_mask < 0] = 0
        confidence_mask *= 255
        confidence_mask = confidence_mask.astype(np.uint8)

        # HWC
        # confidence_mask[np.where(cond)]
        _, confidence_mask = cv2.threshold(
            confidence_mask, int(255 * confidence_thresh), 255, 0)
        contours, hierarchy = cv2.findContours(confidence_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        # print('len(contours)', len(contours))
        if len(contours) == 0:
            return None
        area_max = 0
        # cx_result = None
        # cy_result = None
        box = None
        for i, cnt in enumerate(contours):
            # M = cv2.moments(cnt)
            try:
                # cx = int(M['m10'] / M['m00'])
                # cy = int(M['m01'] / M['m00'])
                area = cv2.contourArea(cnt)
                print('area', area)
                if area < area_thresh:
                    continue
                # print('{} area {} '.format(i, area))
                # if area_max < area:
                #     box = cv2.boundingRect(cnt)
                #     area_max = area

                # else:
                #     continue

                box = cv2.boundingRect(cnt)
                area_max = area

            except Exception:
                continue

            box = expand_roi(box, confidence_mask.shape, scale=1.5)
            if rois_n is None:
                rois_n = torch.tensor(
                    [[box[0], box[1],
                      box[0] + box[2], box[1] + box[3]]],
                    dtype=torch.float32).to('cuda')
            else:
                rois_n = torch.cat((rois_n, torch.tensor(
                    [[box[0], box[1],
                      box[0] + box[2], box[1] + box[3]]],
                    dtype=torch.float32).to('cuda')))

        # if rois is None:
        #     rois = rois_n
        if rois_n is not None:
            rois.append(rois_n)

    return None if rois == [] else rois
    # return cx_result, cy_result, box


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
        # self.n_class = n_class
        self.n_class = 1
        # self.pretrained_net = pretrained_net
        # self.relu = nn.ReLU(inplace=True)
        # self.deconv1 = nn.ConvTranspose2d(
        #     512,
        #     512,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        #     dilation=1,
        #     output_padding=1)
        # self.bn1 = nn.BatchNorm2d(512)
        # self.deconv2 = nn.ConvTranspose2d(
        #     512,
        #     256,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        #     dilation=1,
        #     output_padding=1)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.deconv3 = nn.ConvTranspose2d(
        #     256,
        #     128,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        #     dilation=1,
        #     output_padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.deconv4 = nn.ConvTranspose2d(
        #     128,
        #     64,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        #     dilation=1,
        #     output_padding=1)
        # self.bn4 = nn.BatchNorm2d(64)
        # # self.deconv5 = nn.ConvTranspose2d(
        # #     64,
        # #     32,
        # #     kernel_size=3,
        # #     stride=2,
        # #     padding=1,
        # #     dilation=1,
        # #     output_padding=1)
        # # self.bn5 = nn.BatchNorm2d(32)

        self.conv1 = Conv2DBatchNormRelu(
            512, 128, kernel_size=3, stride=1, padding=1)

        self.deconv2 = nn.ConvTranspose2d(
            128, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = Conv2DBatchNormRelu(
            128, 128, kernel_size=3, stride=1, padding=1)

        # self.deconv3 = nn.ConvTranspose2d(
        #     128, 64, kernel_size=4, stride=2, padding=1)
        # self.conv3 = Conv2DBatchNormRelu(
        #     64, 64, kernel_size=3, stride=1, padding=1)

        self.deconv3 = nn.ConvTranspose2d(
            128, self.n_class, kernel_size=16, stride=8, padding=4)
        self.conv3 = Conv2DBatchNormRelu(
            self.n_class, self.n_class, kernel_size=3, stride=1, padding=1)

        # self.classifier = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.conv2(self.deconv2(h))
        h = self.conv3(self.deconv3(h))
        # h = self.conv4(self.deconv4(h))
        return h
        # output = self.pretrained_net(x)
        # x5 = output['x5']
        # score = self.bn1(self.relu(self.deconv1(x)))
        # score = self.bn2(self.relu(self.deconv2(score)))
        # score = self.bn3(self.relu(self.deconv3(score)))
        # score = self.bn4(self.relu(self.deconv4(score)))
        # score = self.bn5(self.relu(self.deconv5(score)))
        # score = self.classifier(score)

        # return score


class HPNET(nn.Module):
    # def __init__(self, conv_to_head, config):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # config
        # self.feature_stride = config['feature_stride']
        self.feature_compress = config['feature_compress']
        # self.num_feature_channel = config['num_feature_channel']
        # self.num_fc7_channel = config['num_fc7_channel']
        # self.num_rpn_channel = config['num_rpn_channel']
        # self.num_anchor = config['num_anchor']
        # self.score_top_n = config['score_top_n']
        # self.nms_top_n = config['nms_top_n']
        # self.nms_thresh = config['nms_thresh']
        self.pool_out_size = config['pool_out_size']
        self.n_class = config['num_class']

        # self.rois_list = None  # for train
        # layer
        resnet = resnet18()
        self.feature_extractor = nn.Sequential(
            # resnet.conv1,
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

        # fc6
        # self.fc6 = nn.Conv2d(512, 4096, 10)
        # self.relu6 = nn.ReLU(inplace=True)
        # self.drop6 = nn.Dropout2d()

        # fc7
        # self.fc7 = nn.Conv2d(4096, 4096, 1)
        # self.relu7 = nn.ReLU(inplace=True)
        # self.drop7 = nn.Dropout2d()

        # self.score_fr = nn.Conv2d(4096, self.n_class, 1)
        # self.score_fr = nn.Conv2d(512, self.n_class, 1)
        # self.upscore = nn.ConvTranspose2d(self.n_class,
        #                                   self.n_class, 64, stride=32,
        #                                   bias=False)
        # self.upscore = nn.ConvTranspose2d(self.n_class,
        #                                   self.n_class, 64, stride=32,
        #                                   bias=False)

        self.conv_to_head = nn.Sequential(
            # resnet.layer4,
            # resnet.avgpool,
            Conv2DBatchNormRelu(
                512, 128, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128 * 8 * 8),
            nn.Linear(128 * 8 * 8, 5)
        )
        # self.conv_to_head = conv_to_head

        # radios = (0.5, 1, 2), scales = (8, 16, 32)
        # self.radios = config['radios']
        # self.scales = config['scale']

        # self.rpn = RegionProposalNetwork(self.num_feature_channel,
        # self.num_rpn_channel, self.num_anchor)
        # self.roi_pool = RoIAlign(self.pool_out_size, self.feature_compress, -1)
        self.roi_align = RoIAlign(8, 1 / 32, -1)

        # self.classification_head = nn.Linear(
        #     self.num_fc7_channel, self.n_class)
        # self.regression_head = nn.Linear(
        #     self.num_fc7_channel, self.n_class * 4)

    def forward(self, x):
        # torch.backends.cudnn.benchmark = False
        # print('x.shape ', x.shape)
        h = x

        feature = self.feature_extractor(h)
        # pool = self.roi_pool(feature, rois_list)


        # rois_list = torch.cuda.FloatTensor([[0, 10, 10, 100, 100],
        #                                     [1, 10, 10, 100, 200],
        #                                     [1, 10, 10, 100, 300],
        #                                     ])
        # rois_list = torch.cuda.FloatTensor([[10, 10, 100, 100],
        #                                     [10, 10, 100, 200],
        #                                     [10, 10, 100, 300],
                                            # ])

        h = self.decoder(feature)

        # rois_list = [torch.tensor([[10, 10, 100, 100],
        #                           [10, 10, 100, 200],
        #                            [10, 10, 100, 300]], dtype=torch.float32).to(self.device),
        #              torch.tensor([[10, 10, 200, 100],
        #                           [10, 10, 300, 200],
        #                           [10, 10, 400, 300]], dtype=torch.float32).to(self.device)]

        confidence = h[:, 0:1, ...]
        # rois_list = self.find_rois(confidence)
        self.rois_list = find_rois(confidence)
        if self.rois_list is None:
            # set dummy roi
            self.rois_list = [torch.tensor(
                [[0, 0, 0, 0]], dtype=torch.float32).to(self.device)]

        # print('rois_list.shape', rois_list.shape)
        print('rois_list', self.rois_list)

        rois = self.roi_align(feature, self.rois_list)
        # print('rois', rois)
        print('rois.shape', rois.shape)

        depth_and_rotation = self.conv_to_head(rois)
        print('depth_and_rotaion.shape', depth_and_rotation.shape)
        # torch.backends.cudnn.benchmark = True

        # print('pool.shape ', pool.shape)
        # print('depth_and_rotaion.shape', depth_and_rotation.reshape(-1, 5, )shape)

        # h = self.relu6(self.fc6(h))
        # h = self.drop6(h)

        # h = self.relu7(self.fc7(h))
        # h = self.drop7(h)

        # h = self.score_fr(h)
        # h = self.upscore(h)

        return h

        # self.rois_list = rois_list
        # pool = self.roi_pool(feature, rois_list)

        # benchmark because now the input size are fixed
        # torch.backends.cudnn.benchmark = True

        # 4. conv_to_head
        # fc7 = self.conv_to_head(pool)

        # 5. head
        # score = self.classification_head(fc7)
        # bbox = self.regression_head(fc7)
        # return score, bbox.reshape(-1, self.num_class, 4)
