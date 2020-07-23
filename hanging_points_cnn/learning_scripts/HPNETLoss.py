#!/usr/bin/env python
# coding: utf-8

import torch
from torch.nn import Module


def quaternion2matrix(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    m = torch.zeros((3, 3)).to('cuda')
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
        self.Ry = torch.tensor(
            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
            dtype=torch.float32).to('cuda')

    def forward(self, confidence, confidence_gt,
                weight, depth_and_rotation, annotated_rois):

        confidence_diff = confidence[:, 0, ...] - confidence_gt[:, 0, ...]
        confidence_loss = torch.sum(
            weight * confidence_diff ** 2) / (256 ** 2)
        # confidence_loss = torch.sum((weight * confidence_diff) ** 2)

        depth_loss, rotation_loss = torch.tensor(
            0.).to('cuda'), torch.tensor(0.).to('cuda')

        for i, ar in enumerate(annotated_rois):
            if ar[2]:
                depth_loss += (depth_and_rotation[i, 0] - ar[1][0]) ** 2

                m_pred = quaternion2matrix(ar[1][1:])
                q = depth_and_rotation[i, 1:]
                q = q / torch.norm(q)
                m_gt = quaternion2matrix(q)
                rotation_loss += torch.min(
                    torch.norm(m_gt - m_pred),
                    torch.norm(m_gt - m_pred.mm(self.Ry)))

        if len(annotated_rois) > 0:
            depth_loss /= len(annotated_rois)
            rotation_loss /= len(annotated_rois)
        # depth_loss *= 10000
        # rotation_loss *= 10000

        print('confidence_loss', float(confidence_loss))
        # print('depth_loss', float(depth_loss))
        print('rotation_loss', float(rotation_loss))

        return confidence_loss, depth_loss, rotation_loss
