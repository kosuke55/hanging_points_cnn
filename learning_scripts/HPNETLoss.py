#!/usr/bin/env python
# coding: utf-8

import torch
from torch.nn import Module


def quaternion2matrix(q, normalize=False):
    if q.ndim == 1:
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]

        # m = np.zeros((3, 3))
        m = torch.zeros((3, 3))
        m[0, 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
        m[0, 1] = 2 * (q1 * q2 - q0 * q3)
        m[0, 2] = 2 * (q1 * q3 + q0 * q2)

        m[1, 0] = 2 * (q1 * q2 + q0 * q3)
        m[1, 1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
        m[1, 2] = 2 * (q2 * q3 - q0 * q1)

        m[2, 0] = 2 * (q1 * q3 - q0 * q2)
        m[2, 1] = 2 * (q2 * q3 + q0 * q1)
        m[2, 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

    return m


class HPNETLoss(Module):
    def __init__(self):
        super(HPNETLoss, self).__init__()

    def forward(self, output, target, weight, mask):

        confidence_diff = output[:, 0, ...] - target[:, 0, ...]
        confidence_loss = torch.sum((weight * confidence_diff) ** 2)

        # depth_diff = output[:, 1, ...] - target[:, 1, ...]
        # depth_loss = torch.sum((mask * depth_diff) ** 2)

        # rotations_pred = output[:, 2:, ...]
        # print(torch.norm(rotations_pred, dim=1).shape)
        # rn = rotations_pred / torch.norm(rotations_pred, dim=1)[:, None, ...]
        # print(rn.shape)
        # rotations_loss = (torch.norm(
        #     (target[:, 2:, ...] - rotations_pred / torch.norm(
        #         rotations_pred, dim=1)[:, None, ...]) * mask[:, None, ...]))

        print('confidence_loss', float(confidence_loss))
        # print('depth_loss', float(depth_loss))
        # print('rotations_loss', float(rotations_loss))

        # loss = confidence_loss + depth_loss * 100
        # loss = confidence_loss + depth_loss + rotations_loss
        loss = confidence_loss

        return loss
