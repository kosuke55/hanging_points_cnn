#!/usr/bin/env python
# coding: utf-8

import torch
from torch.nn import Module


class UNETLoss(Module):
    def __init__(self):
        super(UNETLoss, self).__init__()

    def forward(self, output, target, weight, mask):
        confidence_diff = output[:, 0, ...] - target[:, 0, ...]
        confidence_loss = torch.sum((weight * confidence_diff) ** 2)

        depth_diff = output[:, 1, ...] - target[:, 1, ...]
        depth_loss = torch.sum((mask * depth_diff) ** 2)

        rotations_pred = output[:, 2:, ...]
        print(torch.norm(rotations_pred, dim=1).shape)
        rn = rotations_pred / torch.norm(rotations_pred, dim=1)[:, None, ...]
        print(rn.shape)
        rotations_loss = (torch.norm(
            (target[:, 2:, ...] - rotations_pred / torch.norm(
                rotations_pred, dim=1)[:, None, ...]) * mask[:, None, ...]))

        print('confidence_loss', float(confidence_loss))
        print('depth_loss', float(depth_loss))
        print('rotations_loss', float(rotations_loss))

        loss = confidence_loss + depth_loss * 100 + rotations_loss * 100
        # loss = confidence_loss + depth_loss

        return loss
