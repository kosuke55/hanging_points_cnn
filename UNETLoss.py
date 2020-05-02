#!/usr/bin/env python
# coding: utf-8

import torch
from torch.nn import Module


class UNETLoss(Module):
    def __init__(self):
        super(UNETLoss, self).__init__()

    def forward(self, output, target, weight):
        diff = output[:, 0, ...] - target[:, 0, ...]
        loss = torch.sum((weight * diff) ** 2)

        return loss
