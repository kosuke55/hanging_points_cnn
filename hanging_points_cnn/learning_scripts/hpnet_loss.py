#!/usr/bin/env python
# coding: utf-8


import numpy as np
import torch
from skrobot import coordinates
from torch.nn import Module


def two_vectors_angle(v1, v2):
    cos = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
    return torch.acos(cos)


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
    def __init__(self, use_coords):
        super(HPNETLoss, self).__init__()
        self.Ry = torch.tensor(
            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
            dtype=torch.float32).to('cuda')
        self.vx = torch.tensor([1., 0, 0], dtype=torch.float32).to('cuda')
        self.use_coords = use_coords

    def forward(self, confidence, confidence_gt,
                weight, depth_and_rotation, annotated_rois):
        sigma = 1.0  # huber
        confidence_diff = confidence[:, 0, ...] - confidence_gt[:, 0, ...]
        confidence_loss = torch.sum(
            weight * torch.where(
                torch.abs(confidence_diff) <= sigma,
                0.5 * (confidence_diff ** 2),
                sigma * torch.abs(confidence_diff) - 0.5 * (sigma ** 2)
            )) / (256 ** 2)
        # confidence_loss = torch.sum((weight * confidence_diff) ** 2)

        depth_loss, rotation_loss = torch.tensor(
            0.).to('cuda'), torch.tensor(0.).to('cuda')

        for i, ar in enumerate(annotated_rois):
            if ar[2]:
                print('dep pred gt', float(depth_and_rotation[i, 0]), ar[1][0])
                print(depth_and_rotation[i, 0], ar[1][0])
                depth_diff = depth_and_rotation[i, 0] - ar[1][0]
                sigma = 0.1  # huber
                depth_loss += 10 * torch.where(
                    torch.abs(depth_diff) <= sigma,
                    0.5 * (depth_diff ** 2),
                    sigma * torch.abs(depth_diff) - 0.5 * (sigma ** 2))
                # depth_loss += (depth_and_rotation[i, 0] - ar[1][0]) ** 2

                # 1 dof
                if self.use_coords:
                    q = depth_and_rotation[i, 1:]
                    q = q / torch.norm(q)
                    m_pred = quaternion2matrix(q)
                    v_pred = torch.matmul(m_pred, self.vx)
                else:
                    v_pred = depth_and_rotation[i, 1:4]
                    v_pred = v_pred / torch.norm(v_pred)

                if torch.any(v_pred == torch.tensor([np.inf] * 3).to('cuda')) \
                        or torch.any(
                            v_pred == torch.tensor([np.nan] * 3).to('cuda')):
                    continue
                m_gt = quaternion2matrix(ar[1][1:])
                v_gt = torch.matmul(m_gt, self.vx)
                rotation_loss += torch.min(
                    two_vectors_angle(v_pred, v_gt),
                    two_vectors_angle(v_pred, -v_gt))

                # 3 dof
                # m_gt = quaternion2matrix(ar[1][1:])
                # q = depth_and_rotation[i, 1:]
                # q = q / torch.norm(q)
                # m_pred = quaternion2matrix(q)
                # rotation_loss += torch.min(
                #     torch.norm(m_gt - m_pred),
                #     torch.norm(m_gt - m_pred.mm(self.Ry)))

        if len(annotated_rois) > 0:
            depth_loss /= len(annotated_rois)
            rotation_loss /= len(annotated_rois)
        # depth_loss *= 10000
        # rotation_loss *= 10000
        print('confidence_diff', float(torch.sum(confidence_diff)))
        print('confidence_loss', float(confidence_loss))
        print('depth_loss', float(depth_loss))
        print('rotation_loss', float(rotation_loss))

        return confidence_loss, depth_loss, rotation_loss
