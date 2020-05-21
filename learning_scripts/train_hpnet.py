#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys
from datetime import datetime

import cameramodels
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import skrobot
import torch
import torch.optim as optim
import visdom

from utils.rois_tools import annotate_rois, find_rois
from HangingPointsData import load_dataset
from HPNETLoss import HPNETLoss
from hpnet import HPNET


def draw_axis(img, R, t, K):
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32(
        [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(
        img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()),
        (0, 0, 255), 2)
    img = cv2.line(
        img,
        tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()),
        (0, 255, 0), 2)
    img = cv2.line(
        img,
        tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()),
        (255, 0, 0), 2)
    return img


def colorize_depth(depth, min_value=None, max_value=None):
    min_value = np.nanmin(depth) if min_value is None else min_value
    max_value = np.nanmax(depth) if max_value is None else max_value

    gray_depth = depth.copy()
    nan_mask = np.isnan(gray_depth)
    gray_depth[nan_mask] = 0
    gray_depth = 255 * (gray_depth - min_value) / (max_value - min_value)
    gray_depth[gray_depth <= 0] = 0
    gray_depth[gray_depth > 255] = 255
    gray_depth = gray_depth.astype(np.uint8)
    colorized = cv2.applyColorMap(gray_depth, cv2.COLORMAP_JET)
    # colorized[nan_mask] = (0, 0, 0)

    return colorized


def create_circular_mask(h, w, cy, cx, radius=50):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = dist_from_center <= radius
    return mask


def create_depth_circle(img, cy, cx, value, radius=50):
    depth_mask = np.zeros_like(img)
    depth_mask[np.where(img == 0)] = 1
    circlular_mask = np.zeros_like(img)
    circlular_mask_idx = np.where(
        create_circular_mask(img.shape[0], img.shape[1], cy, cx,
                             radius=radius))
    circlular_mask[circlular_mask_idx] = 1
    img[circlular_mask_idx] = value


def train(data_path, batch_size, max_epoch, pretrained_model,
          train_data_num, test_data_num, save_dir,
          width=256, height=256):
    intrinsics = np.load(
        os.path.join(data_path, "intrinsics/intrinsics.npy"))
    cameramodel = cameramodels.PinholeCameraModel.from_intrinsic_matrix(
        intrinsics, 1080, 1920)

    train_dataloader, test_dataloader = load_dataset(data_path, batch_size)
    now = datetime.now().strftime('%Y%m%d_%H%M')
    best_loss = 1e10
    vis = visdom.Visdom(port='6006')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    config = {
        'feature_compress': 1 / 16,
        'num_class': 6,
        'pool_out_size': 8,
    }

    hpnet_model = HPNET(config).to(device)
    if os.path.exists(pretrained_model):
        print('use pretrained model')
        hpnet_model.load_state_dict(torch.load(pretrained_model), strict=False)

    # Train
    hpnet_model.train()
    optimizer = optim.SGD(hpnet_model.parameters(), lr=1e-8, momentum=0.9)
    prev_time = datetime.now()
    for epo in range(max_epoch):
        train_loss = 0
        for index, (hp_data, clip_info, hp_data_gt) in enumerate(
                train_dataloader):
            xmin = clip_info[0, 0]
            xmax = clip_info[0, 1]
            ymin = clip_info[0, 2]
            ymax = clip_info[0, 3]

            pos_weight = hp_data_gt.detach().numpy().copy()
            pos_weight = pos_weight[:, 0, ...]
            zeroidx = np.where(pos_weight < 0.5)
            nonzeroidx = np.where(pos_weight >= 0.5)
            pos_weight[zeroidx] = 0.5
            pos_weight[nonzeroidx] = 1.0
            pos_weight = torch.from_numpy(pos_weight)
            pos_weight = pos_weight.to(device)

            criterion = HPNETLoss().to(device)
            hp_data = hp_data.to(device)

            optimizer.zero_grad()

            hp_data_gt = hp_data_gt.to(device)
            ground_truth = hp_data_gt.cpu().detach().numpy().copy()
            confidence_gt = hp_data_gt[:, 0:1, ...]
            rois_list_gt = find_rois(confidence_gt)

            confidence, depth_and_rotation = hpnet_model.forward(hp_data)
            confidence_np = confidence[0, ...].cpu().detach().numpy().copy()
            confidence_np[confidence_np >= 1] = 1.
            confidence_np[confidence_np <= 0] = 0.
            confidence_vis = cv2.cvtColor(confidence_np[0, ...] * 255,
                                          cv2.COLOR_GRAY2BGR)

            depth_and_rotation_gt = hp_data_gt[:, 1:, ...]
            annotated_rois = annotate_rois(
                hpnet_model.rois_list, rois_list_gt, depth_and_rotation_gt)

            loss = criterion(confidence, hp_data_gt, pos_weight,
                             depth_and_rotation, annotated_rois)
            loss.backward()
            iter_loss = loss.item()
            train_loss += iter_loss
            optimizer.step()

            depth = hp_data.cpu().detach().numpy(
            ).copy()[0, 0, ...] * 1000
            depth_bgr = colorize_depth(depth.copy(), 100, 1500)
            depth_rgb = cv2.cvtColor(
                depth_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

            hanging_point_depth_gt \
                = ground_truth[:, 1, ...].astype(np.float32) * 1000
            rotations_gt = ground_truth[0, 2:, ...]
            rotations_gt = rotations_gt.transpose(1, 2, 0)
            hanging_point_depth_gt_bgr \
                = colorize_depth(hanging_point_depth_gt[0, ...], 100, 1500)
            hanging_point_depth_gt_rgb = cv2.cvtColor(
                hanging_point_depth_gt_bgr,
                cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

            axis_gt = depth_bgr.copy()
            axis_large_gt = np.zeros((1080, 1920, 3))
            axis_large_gt[ymin:ymax, xmin:xmax] \
                = cv2.resize(axis_gt, (xmax - xmin, ymax - ymin))

            confidence_gt_vis = cv2.cvtColor(confidence_gt[0, 0, ...].cpu(
            ).detach().numpy().copy() * 255, cv2.COLOR_GRAY2BGR)

            # Visualize gt axis and roi
            rois_gt_filtered = []
            dep_gt = []
            for roi in rois_list_gt[0]:
                if roi.tolist() == [0, 0, 0, 0]:
                    continue
                roi = roi.cpu().detach().numpy().copy()
                cx = int((roi[0] + roi[2]) / 2)
                cy = int((roi[1] + roi[3]) / 2)

                # expand roiしたらなどしたらdep = 0になる場合はある. そしたら弾きたい.
                dep = hanging_point_depth_gt[0, cy, cx]
                dep_gt.append(dep)

                rotation = rotations_gt[cy, cx, :]
                pixel_point = [int(cx * (xmax - xmin) / float(256) + xmin),
                               int(cy * (ymax - ymin) / float(256) + ymin)]
                hanging_point_pose = np.array(
                    cameramodel.project_pixel_to_3d_ray(
                        pixel_point)) * dep * 0.001

                try:
                    draw_axis(axis_large_gt,
                              skrobot.coordinates.math.quaternion2matrix(
                                  rotation),
                              hanging_point_pose,
                              intrinsics)
                except Exception:
                    print('Fail to draw axis')
                    pass

                rois_gt_filtered.append(roi)
            print('dep_gt', dep_gt)

            axis_gt = cv2.resize(axis_large_gt[ymin:ymax, xmin:xmax],
                                 (256, 256)).astype(np.uint8)
            axis_gt = cv2.cvtColor(
                axis_gt, cv2.COLOR_BGR2RGB)

            # draw rois
            for roi in rois_gt_filtered:
                confidence_gt_vis = cv2.rectangle(
                    confidence_gt_vis, (roi[0], roi[1]), (roi[2], roi[3]),
                    (0, 255, 0), 3)
                axis_gt = cv2.rectangle(
                    axis_gt, (roi[0], roi[1]), (roi[2], roi[3]),
                    (0, 255, 0), 3)

            # Visualize pred axis and roi
            dep_pred = []
            for i, roi in enumerate(hpnet_model.rois_list[0]):
                if roi.tolist() == [0, 0, 0, 0]:
                    continue
                roi = roi.cpu().detach().numpy().copy()
                cx = int((roi[0] + roi[2]) / 2)
                cy = int((roi[1] + roi[3]) / 2)
                dep = depth_and_rotation[i, 0] * 1000
                dep_pred.append(float(dep))
                confidence_vis = cv2.rectangle(
                    confidence_vis, (roi[0], roi[1]), (roi[2], roi[3]),
                    (0, 255, 0), 3)
                create_depth_circle(depth, cy, cx, dep.cpu().detach())

            print('dep_pred', dep_pred)
            depth_pred_bgr = colorize_depth(depth, 100, 1500)
            depth_pred_rgb = cv2.cvtColor(
                depth_pred_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

            if np.mod(index, 1) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(
                    epo,
                    index,
                    len(train_dataloader),
                    iter_loss))
                # vis.images(depth_rgb,
                #            win='depth_rgb',
                #            opts=dict(
                #                titlep='depth_rgb'))
                vis.images([hanging_point_depth_gt_rgb,
                            depth_pred_rgb],
                           win='hanging_point_depth_gt_rgb',
                           opts=dict(
                               title='hanging_point_depth (GT, pred)'))
                vis.images([axis_gt.transpose(2, 0, 1)],
                           # axis.transpose(2, 0, 1)],
                           win='train axis',
                           opts=dict(
                               title='train axis'))
                vis.images([confidence_gt_vis.transpose(2, 0, 1),
                            confidence_vis.transpose(2, 0, 1)],
                           win='train_confidence_roi',
                           opts=dict(
                               title='train confidence(GT, Pred)'))

            if index == train_data_num - 1:
                print("Finish train {} data. So start test.".format(index))
                break

        if len(train_dataloader) > 0:
            avg_train_loss = train_loss / len(train_dataloader)
        else:
            avg_train_loss = train_loss

        test_loss = 0
        hpnet_model.eval()
        print("start test.")
        with torch.no_grad():
            for index, (hp_data, clip_info, hp_data_gt) in enumerate(
                    test_dataloader):
                # TO DO
                pass

            avg_test_loss = test_loss / len(test_dataloader) if len(
                test_dataloader) > 0 else 0

        vis.line(X=np.array([epo]), Y=np.array([avg_train_loss]), win='loss',
                 name='avg_train_loss', update='append')
        vis.line(X=np.array([epo]), Y=np.array([avg_test_loss]), win='loss',
                 name='avg_test_loss', update='append')

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        if np.mod(epo, 10) == 0 and epo > 0:
            torch.save(
                hpnet_model.state_dict(),
                os.path.join(
                    save_dir,
                    'hpnet_latestmodel_' + now + '.pt'))
            print(
                'epoch train loss = %f, epoch test loss = %f, best_loss = %f, %s' %
                (train_loss / len(train_dataloader),
                 test_loss / len(test_dataloader),
                 best_loss,
                 time_str))
            if best_loss > test_loss / len(test_dataloader):
                print('update best model {} -> {}'.format(
                    best_loss, test_loss / len(test_dataloader)))
                best_loss = test_loss / len(test_dataloader)
                torch.save(
                    hpnet_model.state_dict(),
                    os.path.join(
                        save_dir,
                        'hpnet_bestmodel_' + now + '.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--data_path',
        '-dp',
        type=str,
        help='Training data path',
        default='/media/kosuke/SANDISK/meshdata/Hanging-ObjectNet3D-DoubleFaces/rotations_0514_1000')
    # default='/media/kosuke/SANDISK/meshdata/Hanging-ObjectNet3D-DoubleFaces/cup')
    parser.add_argument('--batch_size', '-bs', type=int,
                        help='batch size',
                        default=16)
    parser.add_argument('--max_epoch', '-me', type=int,
                        help='max epoch',
                        default=1000000)
    parser.add_argument(
        '--pretrained_model',
        '-p',
        type=str,
        help='Pretrained model',
        default='/media/kosuke/SANDISK/hanging_points_net/checkpoints/resnet/hpnet_latestmodel_20200521_2300.pt')

    parser.add_argument('--train_data_num', '-tr', type=int,
                        help='How much data to use for training',
                        default=1000000)
    parser.add_argument('--test_data_num', '-te', type=int,
                        help='How much data to use for testing',
                        default=1000000)
    parser.add_argument(
        '--save_dir',
        '-sd',
        type=str,
        help='pt model save dir',
        default='/media/kosuke/SANDISK/hanging_points_net/checkpoints/resnet')

    args = parser.parse_args()
    train(data_path=args.data_path,
          batch_size=args.batch_size,
          max_epoch=args.max_epoch,
          pretrained_model=args.pretrained_model,
          train_data_num=args.train_data_num,
          test_data_num=args.test_data_num,
          save_dir=args.save_dir)
