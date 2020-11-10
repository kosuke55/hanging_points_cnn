#!/usr/bin/env python3
# coding: utf-8

import torch
from torchsummary import summary

from hpnet import HPNET

config = {
    'output_channels': 1,
    'feature_extractor_name': 'resnet50',
    'confidence_thresh': 0.3,
    'depth_range': [100, 1500],
    'use_bgr': True,
    'use_bgr2gray': True,
    'roi_padding': 50
}

pretrained_model = '/media/kosuke55/SANDISK-2/meshdata/random_shape_shapenet_hanging_render/1010/gan_2000per0-1000obj_1020.pt'

model = HPNET(config).cuda()
model.load_state_dict(torch.load(pretrained_model), strict=False)
summary(model, (2, 256, 256))  # summary(model,(channels,H,W))
