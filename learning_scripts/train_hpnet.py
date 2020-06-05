#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import os.path as osp
import sys
from datetime import datetime

import cameramodels
import numpy as np
import torch
import torch.optim as optim
import visdom
from skrobot.coordinates.math import quaternion2matrix

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))  # noqa:
from hpnet import HPNET
from HPNETLoss import HPNETLoss
from HangingPointsData import load_dataset
from utils.visualize import colorize_depth, create_depth_circle, draw_axis
from utils.rois_tools import annotate_rois, find_rois

try:
    import cv2
except ImportError:
    for path in sys.path:
        if '/opt/ros/' in path:
            print('sys.path.remove({})'.format(path))
            sys.path.remove(path)
            import cv2
            sys.path.append(path)


def train(data_path, batch_size, max_epoch, pretrained_model,
          train_data_num, val_data_num, save_dir,
          width=256, height=256):
    intrinsics = np.load(
        os.path.join(data_path, "intrinsics/intrinsics.npy"))
    cameramodel = cameramodels.PinholeCameraModel.from_intrinsic_matrix(
        intrinsics, 1080, 1920)

    train_dataloader, val_dataloader = load_dataset(data_path, batch_size)
    now = datetime.now().strftime('%Y%m%d_%H%M')
    best_loss = 1e10
    vis = visdom.Visdom(port='6006')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    config = {
        'feature_compress': 1 / 16,
        'num_class': 6,
        'pool_out_size': 8,
        'confidence_thresh': 0.3,
    }

    hpnet_model = HPNET(config).to(device)
    if os.path.exists(pretrained_model):
        print('use pretrained model')
        hpnet_model.load_state_dict(torch.load(pretrained_model), strict=False)

    # Train
    hpnet_model.train()
    optimizer = optim.SGD(hpnet_model.parameters(), lr=1e-3, momentum=0.9,
                          weight_decay=1e-6)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                            lr_lambda=lambda epo: 0.9 ** epo)
    prev_time = datetime.now()
    for epo in range(max_epoch):
        train_loss = 0
        confidence_train_loss = 0
        depth_train_loss = 0
        rotation_train_loss = 0
        rotation_train_loss_count = 0

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
            rois_list_gt, rois_center_list_gt = find_rois(confidence_gt)
            # print('rois_list_gt.shape')
            # print(len(rois_list_gt), rois_list_gt[0].shape, rois_list_gt[0])
            # print('rois_center_list_gt.shape')
            # print(len(rois_center_list_gt),
            #       rois_center_list_gt[0].shape, rois_center_list_gt[0])

            confidence, depth_and_rotation = hpnet_model(hp_data)
            confidence_np = confidence[0, ...].cpu().detach().numpy().copy()
            confidence_np[confidence_np >= 1] = 1.
            confidence_np[confidence_np <= 0] = 0.
            confidence_vis = cv2.cvtColor(confidence_np[0, ...] * 255,
                                          cv2.COLOR_GRAY2BGR)

            depth_and_rotation_gt = hp_data_gt[:, 1:, ...]
            if hpnet_model.rois_list is None or rois_list_gt is None:
                continue
            annotated_rois = annotate_rois(
                hpnet_model.rois_list, rois_list_gt, depth_and_rotation_gt)

            confidence_loss, depth_loss, rotation_loss \
                = criterion(confidence, hp_data_gt, pos_weight,
                            depth_and_rotation, annotated_rois)

            loss = confidence_loss * 0.1 + rotation_loss

            # if depth_loss > 0.01:
            #     print('loss function1')
            #     loss = confidence_loss + depth_loss * 10 + rotation_loss
            # elif depth_loss > 0.001:
            #     print('loss function2')
            #     loss = confidence_loss + depth_loss * 1000 + rotation_loss
            # else:
            #     print('loss function3')
            #     loss = confidence_loss + depth_loss * 1000 + rotation_loss * 10

            loss.backward()

            # loss_for_record = confidence_loss + depth_loss + rotation_loss
            # loss_for_record = confidence_loss + rotation_loss
            # iter_loss = loss_for_record.item()
            # train_loss += iter_loss
            confidence_train_loss += confidence_loss.item()
            # depth_train_loss += depth_loss.item()
            if rotation_loss > 0:
                rotation_train_loss += rotation_loss.item()
                train_loss = train_loss + confidence_loss.item() + rotation_loss.item()
                rotation_train_loss_count += 1

            optimizer.step()

            depth = hp_data.cpu().detach().numpy(
            ).copy()[0, 0, ...] * 1000
            depth_bgr = colorize_depth(depth.copy(), 100, 1500)
            # depth_rgb = cv2.cvtColor(
            #     depth_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

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
            for roi, roi_c in zip(rois_list_gt[0], rois_center_list_gt[0]):
                if roi.tolist() == [0, 0, 0, 0]:
                    continue
                roi = roi.cpu().detach().numpy().copy()
                # cx = int((roi[0] + roi[2]) / 2)
                # cy = int((roi[1] + roi[3]) / 2)
                cx = roi_c[0]
                cy = roi_c[1]

                # dep = hanging_point_depth_gt[0, cy, cx]
                dep = depth[int(roi[1]):int(roi[3]),
                            int(roi[0]):int(roi[2])]
                dep = np.median(dep[np.where(
                    np.logical_and(dep > 200, dep < 1000))]).astype(np.uint8)
                dep_gt.append(dep)

                rotation = rotations_gt[cy, cx, :]
                pixel_point = [int(cx * (xmax - xmin) / float(256) + xmin),
                               int(cy * (ymax - ymin) / float(256) + ymin)]
                hanging_point_pose = np.array(
                    cameramodel.project_pixel_to_3d_ray(
                        pixel_point)) * dep * 0.001

                try:
                    draw_axis(axis_large_gt,
                              # skrobot.coordinates.math.quaternion2matrix(
                              quaternion2matrix(rotation),
                              hanging_point_pose,
                              intrinsics)
                except Exception:
                    print('Fail to draw axis')
                    pass

                rois_gt_filtered.append(roi)
            print('dep_gt', dep_gt)

            axis_gt = cv2.resize(
                cv2.cvtColor(
                    axis_large_gt[ymin:ymax, xmin:xmax].astype(
                        np.uint8), cv2.COLOR_BGR2RGB), (256, 256))

            # draw gt rois
            for roi in rois_gt_filtered:
                confidence_gt_vis = cv2.rectangle(
                    confidence_gt_vis, (roi[0], roi[1]), (roi[2], roi[3]),
                    (0, 255, 0), 3)
                axis_gt = cv2.rectangle(
                    axis_gt, (roi[0], roi[1]), (roi[2], roi[3]),
                    (0, 255, 0), 3)

            # Visualize pred axis and roi
            axis_pred = depth_bgr.copy()
            axis_large_pred = np.zeros((1080, 1920, 3))
            axis_large_pred[ymin:ymax, xmin:xmax]\
                = cv2.resize(axis_pred, (xmax - xmin, ymax - ymin))
            dep_pred = []
            for i, roi in enumerate(hpnet_model.rois_list[0]):
                if roi.tolist() == [0, 0, 0, 0]:
                    continue
                roi = roi.cpu().detach().numpy().copy()
                cx = int((roi[0] + roi[2]) / 2)
                cy = int((roi[1] + roi[3]) / 2)
                # dep = depth_and_rotation[i, 0] * 1000
                # dep_pred.append(float(dep))
                dep = depth[int(roi[1]):int(roi[3]),
                            int(roi[0]):int(roi[2])]
                dep = np.median(dep[np.where(
                    np.logical_and(dep > 200, dep < 1000))]).astype(np.uint8)

                confidence_vis = cv2.rectangle(
                    confidence_vis, (roi[0], roi[1]), (roi[2], roi[3]),
                    (0, 255, 0), 3)
                if annotated_rois[i][2]:
                    confidence_vis = cv2.rectangle(
                        confidence_vis, (int(annotated_rois[i][0][0]), int(annotated_rois[i][0][1])), (
                            int(annotated_rois[i][0][2]), int(annotated_rois[i][0][3])), (255, 0, 0), 2)

                # create_depth_circle(depth, cy, cx, dep.cpu().detach())
                create_depth_circle(depth, cy, cx, dep)

                q = depth_and_rotation[i, 1:].cpu().detach().numpy().copy()
                q /= np.linalg.norm(q)
                pixel_point = [int(cx * (xmax - xmin) / float(256

                                                              ) + xmin),
                               int(cy * (ymax - ymin) / float(256) + ymin)]
                hanging_point_pose = np.array(
                    cameramodel.project_pixel_to_3d_ray(
                        pixel_point)) * float(dep * 0.001)
                # hanging_point_pose = np.array(
                #     cameramodel.project_pixel_to_3d_ray(
                #         pixel_point)) * float(dep)
                try:
                    draw_axis(axis_large_pred,
                              # skrobot.coordinates.math.quaternion2matrix(
                              quaternion2matrix(q),
                              hanging_point_pose,
                              intrinsics)
                except Exception:
                    print('Fail to draw axis')
                    pass

            # print('dep_pred', dep_pred)

            axis_pred = cv2.resize(axis_large_pred[ymin:ymax, xmin:xmax],
                                   (256, 256)).astype(np.uint8)
            axis_pred = cv2.cvtColor(
                axis_pred, cv2.COLOR_BGR2RGB)

            depth_pred_bgr = colorize_depth(depth, 100, 1500)
            depth_pred_rgb = cv2.cvtColor(
                depth_pred_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

            # draw pred rois
            for i, roi in enumerate(hpnet_model.rois_list[0]):
                if roi.tolist() == [0, 0, 0, 0]:
                    continue
                axis_pred = cv2.rectangle(
                    axis_pred, (roi[0], roi[1]), (roi[2], roi[3]),
                    (0, 255, 0), 3)
                if annotated_rois[i][2]:
                    axis_pred = cv2.rectangle(
                        axis_pred, (int(annotated_rois[i][0][0]), int(annotated_rois[i][0][1])), (
                            int(annotated_rois[i][0][2]), int(annotated_rois[i][0][3])), (255, 0, 0), 2)

            if np.mod(index, 1) == 0:
                print('epoch {}, {}/{},train loss is confidence:{} rotation:{}'.format(
                    epo,
                    index,
                    len(train_dataloader),
                    confidence_train_loss,
                    rotation_train_loss
                ))
                # iter_loss))
                vis.images([hanging_point_depth_gt_rgb,
                            depth_pred_rgb],
                           win='hanging_point_depth_gt_rgb',
                           opts=dict(
                               title='hanging_point_depth (GT, pred)'))
                vis.images([axis_gt.transpose(2, 0, 1),
                            axis_pred.transpose(2, 0, 1)],
                           win='train axis',
                           opts=dict(
                               title='train axis'))
                vis.images([confidence_gt_vis.transpose(2, 0, 1),
                            confidence_vis.transpose(2, 0, 1)],
                           win='train_confidence_roi',
                           opts=dict(
                               title='train confidence(GT, Pred)'))

            if index == train_data_num - 1:
                print("Finish train {} data. So start validation.".format(index))
                break

        if len(train_dataloader) > 0:
            # avg_train_loss = train_loss / len(train_dataloader)
            avg_confidence_train_loss\
                = confidence_train_loss / len(train_dataloader)
            # avg_depth_trannin_loss \
            #     = depth_train_loss / len(train_dataloader)
            if rotation_train_loss_count > 0:
                avg_rotation_train_loss\
                    = rotation_train_loss / rotation_train_loss_count
                avg_train_loss\
                    = train_loss / rotation_train_loss_count
                # train_loss = train_loss + confidence_loss.item() + rotation_loss.item()
                # rotation_train_loss_count += 1
            else:
                avg_rotation_train_loss = rotation_train_loss

        else:
            # avg_train_loss = train_loss
            avg_confidence_train_loss = confidence_train_loss
            # avg_depth_train_loss = depth_train_loss
            avg_rotation_train_loss = rotation_train_loss

        # vis.line(X=np.array([epo]), Y=np.array([avg_train_loss]),
        #          win='loss', name='avg_train_loss', update='append')
        vis.line(X=np.array([epo]), Y=np.array([avg_confidence_train_loss]),
                 win='loss', name='avg_confidence_train_loss', update='append')
        # vis.line(X=np.array([epo]), Y=np.array([avg_depth_train_loss]),
        #          win='loss', name='avg_depth_train_loss', update='append')
        if rotation_train_loss_count > 0:
            vis.line(X=np.array([epo]), Y=np.array([avg_rotation_train_loss]),
                     win='loss', name='rotation_train_loss', update='append')
            vis.line(X=np.array([epo]), Y=np.array([avg_train_loss]),
                     win='loss', name='train_loss', update='append')

        scheduler.step()

        # validation
        val_loss = 0
        confidence_val_loss = 0
        rotation_val_loss = 0
        rotation_val_loss_count = 0
        hpnet_model.eval()
        print("start val.")
        with torch.no_grad():
            for index, (hp_data, clip_info, hp_data_gt) in enumerate(
                    val_dataloader):

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

                # optimizer.zero_grad()

                hp_data_gt = hp_data_gt.to(device)
                ground_truth = hp_data_gt.cpu().detach().numpy().copy()
                confidence_gt = hp_data_gt[:, 0:1, ...]
                rois_list_gt, rois_center_list_gt = find_rois(confidence_gt)

                confidence, depth_and_rotation = hpnet_model(hp_data)
                confidence_np = confidence[0, ...].cpu(
                ).detach().numpy().copy()
                confidence_np[confidence_np >= 1] = 1.
                confidence_np[confidence_np <= 0] = 0.
                confidence_vis = cv2.cvtColor(confidence_np[0, ...] * 255,
                                              cv2.COLOR_GRAY2BGR)

                depth_and_rotation_gt = hp_data_gt[:, 1:, ...]
                if hpnet_model.rois_list is None or rois_list_gt is None:
                    continue
                annotated_rois = annotate_rois(
                    hpnet_model.rois_list, rois_list_gt, depth_and_rotation_gt)

                confidence_loss, depth_loss, rotation_loss\
                    = criterion(confidence, hp_data_gt, pos_weight,
                                depth_and_rotation, annotated_rois)
                # confidence_loss *= 1
                # depth_loss *= 100
                # rotation_loss *= 1

                # loss = confidence_loss + rotation_loss
                confidence_val_loss += confidence_loss.item()
                if rotation_loss > 0:
                    rotation_val_loss += rotation_loss.item()
                    val_loss = val_loss + confidence_loss.item() + rotation_loss.item()
                    rotation_val_loss_count += 1
                # if depth_loss > 0.01:
                #     print('loss function1')
                #     loss = confidence_loss + depth_loss * 10 + rotation_loss
                # elif depth_loss > 0.001:
                #     print('loss function2')
                #     loss = confidence_loss + depth_loss * 1000 + rotation_loss
                # else:
                #     print('loss function3')
                #     loss = confidence_loss + depth_loss * 1000 + rotation_loss * 10

                # loss_for_record = confidence_loss + depth_loss + rotation_loss
                # loss_for_record = confidence_loss + rotation_loss
                # iter_loss = loss_for_record.item()
                # val_loss += iter_loss

                depth = hp_data.cpu().detach().numpy(
                ).copy()[0, 0, ...] * 1000
                depth_bgr = colorize_depth(depth.copy(), 100, 1500)
                # depth_rgb = cv2.cvtColor(
                #     depth_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

                hanging_point_depth_gt\
                    = ground_truth[:, 1, ...].astype(np.float32) * 1000
                rotations_gt = ground_truth[0, 2:, ...]
                rotations_gt = rotations_gt.transpose(1, 2, 0)
                hanging_point_depth_gt_bgr\
                    = colorize_depth(hanging_point_depth_gt[0, ...], 100, 1500)
                hanging_point_depth_gt_rgb = cv2.cvtColor(
                    hanging_point_depth_gt_bgr,
                    cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

                axis_gt = depth_bgr.copy()
                axis_large_gt = np.zeros((1080, 1920, 3))
                axis_large_gt[ymin:ymax, xmin:xmax]\
                    = cv2.resize(axis_gt, (xmax - xmin, ymax - ymin))

                confidence_gt_vis = cv2.cvtColor(confidence_gt[0, 0, ...].cpu(
                ).detach().numpy().copy() * 255, cv2.COLOR_GRAY2BGR)

                # Visualize gt axis and roi
                rois_gt_filtered = []
                dep_gt = []
                for roi, roi_c in zip(rois_list_gt[0], rois_center_list_gt[0]):
                    if roi.tolist() == [0, 0, 0, 0]:
                        continue
                    roi = roi.cpu().detach().numpy().copy()
                    # cx = int((roi[0] + roi[2]) / 2)
                    # cy = int((roi[1] + roi[3]) / 2)
                    cx = roi_c[0]
                    cy = roi_c[1]

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
                                  # skrobot.coordinates.math.quaternion2matrix(
                                  quaternion2matrix(rotation),
                                  hanging_point_pose,
                                  intrinsics)
                    except Exception:
                        print('Fail to draw axis')
                        pass

                    rois_gt_filtered.append(roi)

                axis_gt = cv2.resize(
                    cv2.cvtColor(
                        axis_large_gt[ymin:ymax, xmin:xmax].astype(
                            np.uint8), cv2.COLOR_BGR2RGB), (256, 256))

                # draw gt rois
                for roi in rois_gt_filtered:
                    confidence_gt_vis = cv2.rectangle(
                        confidence_gt_vis, (roi[0], roi[1]), (roi[2], roi[3]),
                        (0, 255, 0), 3)
                    axis_gt = cv2.rectangle(
                        axis_gt, (roi[0], roi[1]), (roi[2], roi[3]),
                        (0, 255, 0), 3)

                # Visualize pred axis and roi
                axis_pred = depth_bgr.copy()
                axis_large_pred = np.zeros((1080, 1920, 3))
                axis_large_pred[ymin:ymax, xmin:xmax]\
                    = cv2.resize(axis_pred, (xmax - xmin, ymax - ymin))
                dep_pred = []
                for i, roi in enumerate(hpnet_model.rois_list[0]):
                    if roi.tolist() == [0, 0, 0, 0]:
                        continue
                    roi = roi.cpu().detach().numpy().copy()
                    cx = int((roi[0] + roi[2]) / 2)
                    cy = int((roi[1] + roi[3]) / 2)

                    # Use pred value
                    # dep = depth_and_rotation[i, 0] * 1000

                    dep = depth[int(roi[1]):int(roi[3]),
                                int(roi[0]):int(roi[2])]
                    dep = np.median(dep[np.where(
                        np.logical_and(dep > 200, dep < 1000))]).astype(np.uint8)
                    # print(dep)xo

                    dep_pred.append(float(dep))
                    confidence_vis = cv2.rectangle(
                        confidence_vis, (roi[0], roi[1]), (roi[2], roi[3]),
                        (0, 255, 0), 3)
                    if annotated_rois[i][2]:
                        confidence_vis = cv2.rectangle(
                            confidence_vis, (int(annotated_rois[i][0][0]), int(annotated_rois[i][0][1])), (
                                int(annotated_rois[i][0][2]), int(annotated_rois[i][0][3])), (255, 0, 0), 2)
                    # create_depth_circle(depth, cy, cx, dep.cpu().detach())
                    create_depth_circle(depth, cy, cx, dep)

                    q = depth_and_rotation[i, 1:].cpu().detach().numpy().copy()
                    q /= np.linalg.norm(q)
                    pixel_point = [int(cx * (xmax - xmin) / float(256) + xmin),
                                   int(cy * (ymax - ymin) / float(256) + ymin)]
                    hanging_point_pose = np.array(
                        cameramodel.project_pixel_to_3d_ray(
                            pixel_point)) * float(dep * 0.001)
                    # hanging_point_pose = np.array(
                    #     cameramodel.project_pixel_to_3d_ray(
                    #         pixel_point)) * float(dep)
                    try:
                        draw_axis(axis_large_pred,
                                  # skrobot.coordinates.math.quaternion2matrix(
                                  quaternion2matrix(q),
                                  hanging_point_pose,
                                  intrinsics)
                    except Exception:
                        print('Fail to draw axis')
                        pass

                print('dep_pred', dep_pred)

                axis_pred = cv2.resize(axis_large_pred[ymin:ymax, xmin:xmax],
                                       (256, 256)).astype(np.uint8)
                axis_pred = cv2.cvtColor(
                    axis_pred, cv2.COLOR_BGR2RGB)

                depth_pred_bgr = colorize_depth(depth, 100, 1500)
                depth_pred_rgb = cv2.cvtColor(
                    depth_pred_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

                # draw pred rois
                for i, roi in enumerate(hpnet_model.rois_list[0]):
                    if roi.tolist() == [0, 0, 0, 0]:
                        continue
                    axis_pred = cv2.rectangle(
                        axis_pred, (roi[0], roi[1]), (roi[2], roi[3]),
                        (0, 255, 0), 3)
                    if annotated_rois[i][2]:
                        axis_pred = cv2.rectangle(
                            axis_pred, (int(annotated_rois[i][0][0]), int(annotated_rois[i][0][1])), (
                                int(annotated_rois[i][0][2]), int(annotated_rois[i][0][3])), (255, 0, 0), 2)

                if np.mod(index, 1) == 0:
                    print('epoch {}, {}/{},val loss is confidence:{} rotation:{}'.format(
                        epo,
                        index,
                        len(val_dataloader),
                        confidence_val_loss,
                        rotation_val_loss
                    ))
                    vis.images([hanging_point_depth_gt_rgb,
                                depth_pred_rgb],
                               win='val hanging_point_depth_gt_rgb',
                               opts=dict(
                                   title='val hanging_point_depth (GT, pred)'))
                    vis.images([axis_gt.transpose(2, 0, 1),
                                axis_pred.transpose(2, 0, 1)],
                               win='val axis',
                               opts=dict(
                                   title='val axis'))
                    vis.images([confidence_gt_vis.transpose(2, 0, 1),
                                confidence_vis.transpose(2, 0, 1)],
                               win='val_confidence_roi',
                               opts=dict(
                                   title='val confidence(GT, Pred)'))

                if index == val_data_num - 1:
                    print("Finish val {} data. So start val.".format(index))
                    break

        if len(val_dataloader) > 0:
            # avg_val_loss = val_loss / len(val_dataloader)
            avg_confidence_val_loss\
                = confidence_val_loss / len(val_dataloader)
            if rotation_val_loss_count > 0:
                avg_rotation_val_loss\
                    = rotation_val_loss / rotation_val_loss_count
                avg_val_loss\
                    = val_loss / rotation_val_loss_count
            else:
                avg_rotation_val_loss = rotation_val_loss
        else:
            # avg_val_loss = val_loss
            avg_confidence_val_loss = confidence_val_loss
            avg_rotation_val_loss = rotation_val_loss
        # vis.line(X=np.array([epo]), Y=np.array([avg_val_loss]), win='loss',
        #          name='avg_val_loss', update='append')
        vis.line(X=np.array([epo]), Y=np.array([avg_confidence_val_loss]),
                 win='loss', name='confidence_val_loss', update='append')
        if rotation_val_loss_count > 0:
            vis.line(X=np.array([epo]), Y=np.array([avg_rotation_val_loss]),
                     win='loss', name='rotation_val_loss', update='append')
            vis.line(X=np.array([epo]), Y=np.array([avg_val_loss]),
                     win='loss', name='val_loss', update='append')

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('----save latest model  2-----')
        torch.save(hpnet_model.state_dict(),
                   os.path.join(save_dir, 'hpnet_latestmodel_' + now + '.pt'))
        if np.mod(epo, 1) == 0 and epo > 1:
            print('----save latest model-----')
            torch.save(hpnet_model.state_dict(),
                       os.path.join(save_dir, 'hpnet_latestmodel_' + now + '.pt'))
        print('epoch val loss = %f, epoch val loss = %f, best_loss = %f, %s' %
              (val_loss / len(val_dataloader),
               val_loss / len(val_dataloader),
               best_loss,
               time_str))
        if rotation_val_loss_count > 0:
            if best_loss > val_loss / rotation_val_loss_count and epo > 1:
                print('update best model {} -> {}'.format(
                    best_loss, val_loss / rotation_val_loss_count))
                best_loss = val_loss / rotation_val_loss_count
                torch.save(
                    hpnet_model.state_dict(),
                    os.path.join(save_dir, 'hpnet_bestmodel_' + now + '.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--data_path',
        '-dp',
        type=str,
        help='Training data path',
        default='/media/kosuke/SANDISK/meshdata/ycb_hanging_object/per2000')
    # default='/media/kosuke/SANDISK/meshdata/Hanging-ObjectNet3D-DoubleFaces/all_0527')
    # default='/media/kosuke/SANDISK/meshdata/Hanging-ObjectNet3D-DoubleFaces/cup')
    parser.add_argument('--batch_size', '-bs', type=int,
                        help='batch size',
                        default=32)
    parser.add_argument('--max_epoch', '-me', type=int,
                        help='max epoch',
                        default=1000000)
    parser.add_argument(
        '--pretrained_model',
        '-p',
        type=str,
        help='Pretrained model',
        default='/media/kosuke/SANDISK/hanging_points_net/checkpoints/resnet/hpnet_latestmodel_20200605_0010.pt')
    # default='/media/kosuke/SANDISK/hanging_points_net/checkpoints/resnet/hpnet_bestmodel_20200527_2110.pt')
    # default='/media/kosuke/SANDISK/hanging_points_net/checkpoints/resnet/hpnet_bestmodel_20200527_1846.pt')
    # '/media/kosuke/SANDISK/hanging_points_net/checkpoints/resnet/hpnet_latestmodel_20200522_0149_.pt')

    parser.add_argument('--train_data_num', '-tr', type=int,
                        help='How much data to use for training',
                        default=1000000)
    parser.add_argument('--val_data_num', '-te', type=int,
                        help='How much data to use for validation',
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
          val_data_num=args.val_data_num,
          save_dir=args.save_dir)
