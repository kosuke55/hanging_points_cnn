#!/usr/bin/env python3
# coding: utf-8

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_functionq

import sys

import numpy as np
import torch
import torchvision

try:
    import cv2
except ImportError:
    for path in sys.path:
        if '/opt/ros/' in path:
            print('sys.path.remove({})'.format(path))
            sys.path.remove(path)
            import cv2
            sys.path.append(path)


def annotate_rois(rois_list, rois_list_gt, feature, iou_thresh=0.3):
    """rois_listに対して正しいgt_roiをiouを計算することによって見つける.

    Parametersa
    ----------
    rois_list : list[torch.tensor]
        NCHW
    rois_list_gt : list[torch.tensor]
        NCHW
    Returns
        rois_pair: list[pred_roi_list[], gt_roi_list[], bool, max_iou]
        len(rois_pair) = Total of all rois for all batches

    listを返す.
        gtがない場合もわかるように
    -------
    """

    if not(len(rois_list) == len(rois_list_gt)):
        print('len(rois_list), len(rois_list_gt',
              len(rois_list), len(rois_list_gt))
        raise

    annotated_rois = []
    for n, (pred, gt) in enumerate(zip(rois_list, rois_list_gt)):
        iou = torchvision.ops.box_iou(gt, pred)
        iou[torch.isnan(iou)] = 0.
        max_iou, max_iou_index = iou.max(dim=0)

        for i, p in enumerate(pred.tolist()):
            try:
                roi_gt = gt.tolist()[max_iou_index.tolist()[i]]
                val_gt = get_val_in_roi(roi_gt, feature[n, ...])
                annotated_rois.append([p,
                                       val_gt.tolist(),
                                       max_iou.tolist()[i] > iou_thresh,
                                       max_iou.tolist()[i]])
            except Exception:
                print(gt.shape)
                print(pred.shape)
                print(max_iou.shape)
                print(max_iou_index.shape)
                raise

    return annotated_rois


def get_val_in_roi(roi, feature):
    """roiに対して中心の値を返す.  depthとrotationを求めたい.

    Parametersa
    ----------
         roi: list

    Returns
    -------

    -------
    """

    cx = int((roi[0] + roi[2]) / 2)
    cy = int((roi[1] + roi[3]) / 2)

    return feature[:, cy, cx]


def reshape_rois_list(rois_list):
    """reshape_rois_list
    rois_listのshapeを
    [[[  8.,  11.,  47.,  96.],
      [ 22.,  48.,  74.,  88.],
      [ 22.,  48.,  74.,  88.],
      [ 62.,  76.,  80., 105.]],

      [[ 24.,  95., 124., 126.],
       [ 14.,  29.,  43., 120.],
       [ 11.,  32.,  35., 121.]]]
    のようにする.

    Parametersa
    ----------
    rois_list : list[torch.tensor]

    Returns
    -------
    reshaped_rois_list : list

    """
    reshaped_rois_list = None
    for i, rl in enumerate(rois_list):
        if reshaped_rois_list is None:
            reshaped_rois_list = rl.detach().numpy().copy()[None, ...].tolist()
        else:
            reshaped_rois_list += rl.detach().numpy().copy()[
                None, ...].tolist()

    return reshaped_rois_list


def expand_box(box, img_shape, scale=1.5):
    """Expand roi box.
    """

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

    return [int(xo), int(yo), int(wo), int(ho)]


def find_rois(confidence,
              confidence_gt=None,
              confidence_thresh=0.5,
              area_thresh=100):
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

    confidence = confidence.cpu().detach().numpy().copy()

    rois = []
    rois_center = []
    for n in range(confidence.shape[0]):
        rois_n = None
        confidence_mask = confidence[n, ...].transpose(1, 2, 0)
        confidence_mask[confidence_mask > 1] = 1
        confidence_mask[confidence_mask < 0] = 0
        confidence_mask *= 255
        confidence_mask = confidence_mask.astype(np.uint8)
        _, confidence_mask = cv2.threshold(
            confidence_mask, int(255 * confidence_thresh), 255, 0)

        if sys.version_info[0] == 2:
            _, contours, hierarchy = cv2.findContours(confidence_mask,
                                                      cv2.RETR_TREE,
                                                      cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv2.findContours(confidence_mask,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            # set dummy rois. None にするとroi_alignでerrorが起きる.
            rois_n = torch.tensor(
                [[0, 0, 0, 0]], dtype=torch.float32).to('cuda')
            try:
                rois.append(rois_n)
                rois_center.append(np.array([0, 0]))

            except Exception:
                print('rois_n', rois_n)
                raise
            continue

        box = None

        for i, cnt in enumerate(contours):
            try:
                area = cv2.contourArea(cnt)
                if area < area_thresh:
                    continue
                box = cv2.boundingRect(cnt)

            except Exception:
                continue

            box_center = [int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)]
            box = expand_box(box, confidence_mask.shape, scale=2.5)
            if rois_n is None:
                rois_n = torch.tensor(
                    [[box[0], box[1],
                      box[0] + box[2], box[1] + box[3]]],
                    dtype=torch.float32).to('cuda')
                rois_n_c = np.array([box_center])

            else:
                rois_n = torch.cat((rois_n, torch.tensor(
                    [[box[0], box[1],
                      box[0] + box[2], box[1] + box[3]]],
                    dtype=torch.float32).to('cuda')))
                rois_n_c = np.concatenate([rois_n_c, [box_center]])

        if rois_n is None:
            rois_n = torch.tensor(
                [[0, 0, 0, 0]], dtype=torch.float32).to('cuda')
            rois_n_c = np.array([[0, 0]])
        try:
            rois.append(rois_n)
            rois_center.append(rois_n_c)
        except Exception:
            print('rois_n', rois_n)
            print('rois_n_c', rois_n_c)
            raise
    # return None, None
    return (None, None) if rois == [] else (rois, rois_center)
