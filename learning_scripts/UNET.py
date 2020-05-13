#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn


class Conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, bias=True):
        super(Conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)
        return outputs


class UNET(nn.Module):
    def __init__(self, in_channels=6):
        super(UNET, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # conv
        self.conv0_1 = Conv2DBatchNormRelu(
            in_channels, 24, kernel_size=1, stride=1, padding=0)
        self.conv0 = Conv2DBatchNormRelu(
            24, 24, kernel_size=3, stride=1, padding=1)

        self.conv1_1 = Conv2DBatchNormRelu(
            24, 48, kernel_size=3, stride=2, padding=1)
        self.conv1 = Conv2DBatchNormRelu(
            48, 48, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = Conv2DBatchNormRelu(
            48, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = Conv2DBatchNormRelu(
            64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2DBatchNormRelu(
            64, 64, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = Conv2DBatchNormRelu(
            64, 96, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = Conv2DBatchNormRelu(
            96, 96, kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv2DBatchNormRelu(
            96, 96, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = Conv2DBatchNormRelu(
            96, 128, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = Conv2DBatchNormRelu(
            128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv2DBatchNormRelu(
            128, 128, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = Conv2DBatchNormRelu(
            128, 192, kernel_size=3, stride=2, padding=1)
        self.conv5 = Conv2DBatchNormRelu(
            192, 192, kernel_size=3, stride=1, padding=1)

        # deconv
        self.deconv5_1 = Conv2DBatchNormRelu(
            192, 192, kernel_size=3, stride=1, padding=1)

        self.deconv4 = nn.ConvTranspose2d(
            192, 128, kernel_size=4, stride=2, padding=1)
        self.deconv4_1 = Conv2DBatchNormRelu(
            256, 128, kernel_size=3, stride=1, padding=1)

        self.deconv3 = nn.ConvTranspose2d(
            128, 96, kernel_size=4, stride=2, padding=1)
        self.deconv3_1 = Conv2DBatchNormRelu(
            192, 96, kernel_size=3, stride=1, padding=1)

        self.deconv2 = nn.ConvTranspose2d(
            96, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2_1 = Conv2DBatchNormRelu(
            128, 64, kernel_size=3, stride=1, padding=1)

        self.deconv1 = nn.ConvTranspose2d(
            64, 48, kernel_size=4, stride=2, padding=1)
        self.deconv1_1 = Conv2DBatchNormRelu(
            96, 48, kernel_size=3, stride=1, padding=1)

        self.deconv0 = nn.ConvTranspose2d(
            48, 6, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # conv
        conv0 = self.conv0(self.conv0_1(x))
        conv1 = self.conv1(self.conv1_1(conv0))
        conv2 = self.conv2(self.conv2_2(self.conv2_1(conv1)))
        conv3 = self.conv3(self.conv3_2(self.conv3_1(conv2)))
        conv4 = self.conv4(self.conv4_2(self.conv4_1(conv3)))
        conv5 = self.conv5(self.conv5_1(conv4))

        # deconv
        deconv5_1 = self.deconv5_1(conv5)

        deconv4 = self.deconv4(deconv5_1)
        concat4 = torch.cat([conv4, deconv4], dim=1)
        deconv4_1 = self.deconv4_1(concat4)

        deconv3 = self.deconv3(deconv4_1)
        concat3 = torch.cat([conv3, deconv3], dim=1)
        deconv3_1 = self.deconv3_1(concat3)

        deconv2 = self.deconv2(deconv3_1)
        concat2 = torch.cat([conv2, deconv2], dim=1)
        deconv2_1 = self.deconv2_1(concat2)

        deconv1 = self.deconv1(deconv2_1)
        concat1 = torch.cat([conv1, deconv1], dim=1)
        deconv1_1 = self.deconv1_1(concat1)

        deconv0 = self.deconv0(deconv1_1)

        return deconv0
