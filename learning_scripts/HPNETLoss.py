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

    # def forward(self, output, target, weight, mask):
    # def forward(self, output, target, weight):
    def forward(self, confidence, confidence_gt,
                weight, depth_and_rotation, annotated_rois):

        confidence_diff = confidence[:, 0, ...] - confidence_gt[:, 0, ...]
        confidence_loss = torch.sum((weight * confidence_diff) ** 2)

        depth_loss, rotation_loss = torch.tensor(0.).to('cuda'), torch.tensor(0.).to('cuda')
        # idx = 0
        # print('len(annotated_rois)', len(annotated_rois))
        for i, ar in enumerate(annotated_rois):
            # print('len(ar_n)', len(ar_n))
            if ar[2]:
                depth_loss += (depth_and_rotation[i, 0] - ar[1][0]) ** 2

            # ar: [[pred_rois], [depth], iou > iou_thresh, max_iou]

            # for _, ar in enumerate(ar_n):
            #     print('ar', ar)
            #     print(ar[2])
            #     if ar[2]:
                #     depth_loss += (depth_and_rotation[idx, 0] - ar[1]) ** 2
                # idx += 1

        # depth_loss  = (depth_and_rotation[:, 0, ...] - annotated_rois[:, 0, ...]) ** 2

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
        print('depth_loss', float(depth_loss))
        # print('depth_loss', float(depth_loss))
        # print('rotations_loss', float(rotations_loss))


        # loss = confidence_loss + depth_loss + rotations_loss
        loss = confidence_loss + depth_loss * 1000
        # loss = confidence_loss

        return loss
