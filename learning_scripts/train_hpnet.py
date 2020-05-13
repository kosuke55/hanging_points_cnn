#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cameramodels
import numpy as np
import skrobot
import torch
import torch.optim as optim
from torchvision import transforms
import visdom
from datetime import datetime

from UNET import UNET
from UNETLoss import UNETLoss
from HangingPointsData import load_dataset


def draw_axis(img, R, t, K):
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32(
        [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(
        img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()),
        (0, 0, 255), 1)
    img = cv2.line(
        img,
        tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()),
        (0, 255, 0), 1)
    img = cv2.line(
        img,
        tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()),
        (255, 0, 0), 1)
    return img


def find_contour_center(img):
    ret, thresh = cv2.threshold(img.copy(), int(255 * 0.5), 255, 0)
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(img, 1, 2)
    print(len(contours))
    if len(contours) == 0:
        return None, None
    cx_max = 0
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if cx_max < cx:
            cx_max = cx
            cx_result = cx
            cy_result = cy

    return cx_result, cy_result


def colorize_depth(depth, min_value=None, max_value=None):
    min_value = np.nanmin(depth) if min_value is None else min_value
    max_value = np.nanmax(depth) if max_value is None else max_value

    gray_depth = depth.copy()
    nan_mask = np.isnan(gray_depth)
    # print(nan_mask)
    gray_depth[nan_mask] = 0
    gray_depth = 255 * (gray_depth - min_value) / (max_value - min_value)
    gray_depth[gray_depth <= 0] = 0
    gray_depth[gray_depth > 255] = 255
    gray_depth = gray_depth.astype(np.uint8)
    colorized = cv2.applyColorMap(gray_depth, cv2.COLORMAP_JET)
    # print('nan_mask shape ', nan_mask.shape)
    # colorized[nan_mask] = (0, 0, 0)
    # colorized[nan_mask] = [0, 0, 0]

    return colorized


def train(data_path, batch_size, max_epoch, pretrained_model,
          train_data_num, test_data_num,
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
    # unet_model = UNET(in_channels=3).to(device)
    unet_model = UNET(in_channels=1).to(device)
    if os.path.exists(pretrained_model):
        print('use pretrained model')
        unet_model.load_state_dict(torch.load(pretrained_model), strict=False)
    unet_model.eval()

    transfer_learning = False
    if transfer_learning:
        params_to_update = []
        update_param_names = ["deconv0.weight", "deconv0.bias"]
        for name, param in unet_model.named_parameters():
            if name in update_param_names:
                param.requires_grad = True
                params_to_update.append(param)
                print(name)
            else:
                param.requires_grad = False
        print("-----------")
        print(params_to_update)
        optimizer = optim.SGD(params=params_to_update, lr=1e-5, momentum=0.9)
    else:
        optimizer = optim.SGD(unet_model.parameters(), lr=1e-7, momentum=0.9)

    prev_time = datetime.now()
    for epo in range(max_epoch):
        train_loss = 0
        unet_model.train()
        for index, (hp_data, clip_info, hp_data_gt) in enumerate(train_dataloader):
            xmin = clip_info[0, 0]
            xmax = clip_info[0, 1]
            ymin = clip_info[0, 2]
            ymax = clip_info[0, 3]

            pos_weight = hp_data_gt.detach().numpy().copy()
            pos_weight = pos_weight[:, 0, ...]
            # zeroidx = np.where(pos_weight < 100)
            # nonzeroidx = np.where(pos_weight >= 10)
            zeroidx = np.where(pos_weight < 0.5)
            nonzeroidx = np.where(pos_weight >= 0.5)
            pos_weight[zeroidx] = 0.5
            pos_weight[nonzeroidx] = 1.0
            pos_weight = torch.from_numpy(pos_weight)
            pos_weight = pos_weight.to(device)
            criterion = UNETLoss().to(device)
            hp_data = hp_data.to(device)
            hp_data_gt = hp_data_gt.to(device)
            optimizer.zero_grad()
            output = unet_model(hp_data)

            confidence_thresh = 0.5
            ground_truth = hp_data_gt.cpu().detach().numpy().copy()
            # confidence_gt = ground_truth[0:1, ...]
            # confidence_gt = ground_truth[0, 0, ...]
            confidence_gt = ground_truth[:, 0, ...]
            print('confidence_gt ', confidence_gt.shape)
            confidence_mask = np.zeros_like(
                confidence_gt, dtype=(np.float32))

            # confidence = output[0, 0, :, :]
            confidence = output[:, 0, :, :]
            confidence_np = confidence.cpu().detach().numpy().copy()
            confidence_np[confidence_np >= 1] = 1.
            confidence_np[confidence_np <= 0] = 0.

            confidence_mask[np.where(
                np.logical_and(
                    confidence_gt > confidence_thresh,
                    confidence_np > confidence_thresh))] = 1.
            # confidence_mask = confidence_mask[None, ...]
            # print('confidence_mask ', confidence_mask.shape)
            # print('pos_weight ', pos_weight.shape)

            depth_pred = output[0, 1, :, :].cpu().detach().numpy().copy() * 1000
            depth_pred_bgr = colorize_depth(depth_pred, 100, 1500)
            depth_pred_rgb = cv2.cvtColor(
                depth_pred_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

            # rotations_gt = ground_truth[:, 2:, ...]
            # rotations_gt = rotations_gt.transpose(1, 2, 0)

            print('confidence_mask ', confidence_mask.shape)
            loss = criterion(output, hp_data_gt, pos_weight,
                             torch.from_numpy(
                                 confidence_mask).to(device))

            loss.backward()
            iter_loss = loss.item()
            train_loss += iter_loss
            optimizer.step()

            # bgr = hp_data.cpu().detach().numpy().copy()[0, :3, ...]
            # bgr = bgr.transpose(1, 2, 0)
            # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # rgb = rgb.transpose(2, 0, 1)

            depth = hp_data.cpu().detach().numpy(
            ).copy()[0, 0, ...] * 1000
            depth_bgr = colorize_depth(depth, 100, 1500)
            # print('depth_bgr.shape ', depth_bgr.shape)
            depth_rgb = cv2.cvtColor(
                depth_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

            # depth = hp_data.cpu().detach().numpy().copy()[0, 3:, ...]
            # depth = depth.transpose(1, 2, 0)
            # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
            # depth = depth.transpose(2, 0, 1)

            # ground_truth = hp_data_gt.cpu().detach().numpy().copy()[0, 0, ...].astype(np.uint8)

            hanging_point_depth_gt \
                = ground_truth[:, 1, ...].astype(np.float32) * 1000
            # print('hanging_point_depth_gt  ', hanging_point_depth_gt.shape)
            # hanging_point_depth_gt_np \
            #     = hanging_point_depth_gt.transpose(1, 2, 0)
            rotations_gt = ground_truth[0, 2:, ...]
            rotations_gt = rotations_gt.transpose(1, 2, 0)
            # rotations_gt = rotations_gt.transpose(0, 2, 3, 1)

            # print('hanging_point_depth_gt 2 ', hanging_point_depth_gt_np.shape)
            hanging_point_depth_gt_bgr \
                = colorize_depth(hanging_point_depth_gt[0, ...], 100, 1500)
            hanging_point_depth_gt_rgb = cv2.cvtColor(
                hanging_point_depth_gt_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            # print(rotations_gt_np.shape)

            confidence_binary_gt = (
                confidence_gt[0, ...] * 255).astype(np.uint8)
            print('confidence_binary_gt.shape ',
                  confidence_binary_gt.shape)

            cx, cy = find_contour_center(confidence_binary_gt)

            axis_gt = depth_bgr.copy()
            if cx is not None and cy is not None:

                dep = hanging_point_depth_gt[0, cy, cx]
                print('hanging_point_depth_gt ',
                      hanging_point_depth_gt.shape, hanging_point_depth_gt[0, cy, cx], dep)

                rotation = rotations_gt[cy, cx, :]
                print('rotation ', rotation)
                print(ymax)
                pixel_point = [int(cx * (xmax - xmin) / float(256) + xmin),
                               int(cy * (ymax - ymin) / float(256) + ymin)]
                print(cameramodel.project_pixel_to_3d_ray(pixel_point))
                     
                hanging_point_pose = np.array(
                    cameramodel.project_pixel_to_3d_ray(pixel_point)) * dep * 0.001
                    # hanging_point_depth_gt[0, cy, cx] * 0.001)

                axis_large_gt = np.zeros((1080, 1920, 3))
                axis_large_gt[ymin:ymax, xmin:xmax] \
                    = cv2.resize(axis_gt, (xmax - xmin, ymax - ymin))
                draw_axis(axis_large_gt,
                          skrobot.coordinates.math.quaternion2matrix(rotation),
                          hanging_point_pose,
                          intrinsics)
                axis_gt = cv2.resize(axis_large_gt[ymin:ymax, xmin:xmax],
                                     (256, 256)).astype(np.uint8)
            axis_gt = cv2.cvtColor(
                axis_gt, cv2.COLOR_BGR2RGB)

            rotations_mask = np.zeros((rotations_gt.shape[0],
                                       rotations_gt.shape[1]))
            for i in range(rotations_gt.shape[0]):
                for j in range(rotations_gt.shape[1]):
                    # TO DO. fix [0, 0, 0, 1] -> [1, 0, 0, 0]
                    if np.all(rotations_gt[i, j, :] != [1, 0, 0, 0]):
                        # print(rotations_gt_np[i, j, :])
                        R = skrobot.coordinates.math.quaternion2matrix(
                            rotations_gt[i, j, :])
                        # rotations_mask[i, j, 0] = 255
                        rotations_mask[i, j] = 255

            # rotations_mask_vis = rotations_mask.transpose(2, 0, 1)
            # print('rotations_gt.shape', rotations_gt.shape)
            # print(ground_truth.shape)

            # ground_truth = ground_truth.astype(np.uint8)

            # print(confidence_np.shape)
            # print(ground_truth.shape)
            # print(np.max(ground_truth))
            # print(np.min(ground_truth))

            # print('confidence_gt.shape')
            # print(confidence_gt.shape)
            # print(confidence_np.shape)

            if np.mod(index, 1) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(
                    epo,
                    index,
                    len(train_dataloader),
                    iter_loss))
                vis.images(depth_rgb,
                           win='depth_rgb',
                           opts=dict(
                               title='depth_rgb'))
                vis.images(hanging_point_depth_gt_rgb,
                           win='hanging_point_depth_gt_rgb',
                           opts=dict(
                               title='hanging_point_depth_gt_rgb'))
                vis.images(depth_pred_rgb,
                           win='train depth_pred_rgb',
                           opts=dict(
                               title='train depth_pred_rgb'))
                # vis.images(rotations_mask,
                #            win='rotations_mask',
                #            opts=dict(
                #                title='rotations_mask'))
                # vis.images(confidence_mask.transpose(2, 0, 1),
                vis.images(confidence_mask[0, ...],
                           win='confidence_mask',
                           opts=dict(
                               title='confidence_mask'))
                vis.images(axis_gt.transpose(2, 0, 1),
                           win='train axis',
                           opts=dict(
                               title='train axis'))
                # vis.images(depth,
                #            win='depth',
                #            opts=dict(
                #                title='depth'))

                # vis.images([confidence_gt, confidence_np],
                vis.images([confidence_gt[0, ...][None, ...],
                            confidence_np[0, ...][None, ...]],
                           win='train_confidence',
                           opts=dict(
                               title='train confidence(GT, Pred)'))
                # vis.images([confidence_gt],
                #            win='train_confidence gt',
                #            opts=dict(
                #                title='train confidence(GT)'))
                # vis.images([confidence_np],
                #            win='train_confidence pred',
                #            opts=dict(
                #                title='train confidence(Pred)'))

            if index == train_data_num - 1:
                print("Finish train {} data. So start test.".format(index))
                break

        if len(train_dataloader) > 0:
            avg_train_loss = train_loss / len(train_dataloader)
        else:
            avg_train_loss = train_loss

        test_loss = 0
        unet_model.eval()

        with torch.no_grad():
            for index, (hp_data, clip_info, hp_data_gt) in enumerate(test_dataloader):
                # pos_weight = hp_data_gt.detach().numpy().copy()[0, 0, ...]
                # zeroidx = np.where(pos_weight < 10)
                # nonzeroidx = np.where(pos_weight >= 10)
                pos_weight = hp_data_gt.detach().numpy().copy()
                pos_weight = pos_weight[:, 0, ...]
                zeroidx = np.where(pos_weight < 0.5)
                nonzeroidx = np.where(pos_weight >= 0.5)
                pos_weight[zeroidx] = 0.5
                pos_weight[nonzeroidx] = 1.0
                pos_weight = torch.from_numpy(pos_weight)
                pos_weight = pos_weight.to(device)
                criterion = UNETLoss().to(device)
                hp_data = hp_data.to(device)
                hp_data_gt = hp_data_gt.to(device)
                optimizer.zero_grad()
                output = unet_model(hp_data)

                confidence_thresh = 0.5
                # ground_truth = hp_data_gt.cpu().detach().numpy()[0, ...].copy()
                ground_truth = hp_data_gt.cpu().detach().numpy().copy()
                # confidence_gt = ground_truth[0:1, ...]
                # confidence_gt = ground_truth[0, ...]
                confidence_gt = ground_truth[:, 0, ...]
                confidence_mask = np.zeros_like(confidence_gt)

                # confidence = output[0, 0:1, :, :]
                # confidence = output[0, 0, :, :]
                confidence = output[:, 0, :, :]
                confidence_np = confidence.cpu().detach().numpy().copy()
                confidence_np[confidence_np >= 1] = 1.
                confidence_np[confidence_np <= 0] = 0.

                confidence_mask[np.where(
                    np.logical_and(
                        confidence_gt > confidence_thresh,
                        confidence_np > confidence_thresh))] = 1
                # confidence_mask = confidence_mask[None, ...]

                # loss = criterion(output, hp_data_gt, pos_weight)
                loss = criterion(output, hp_data_gt, pos_weight,
                                 torch.tensor(
                                     confidence_mask).to(device))

                iter_loss = loss.item()
                test_loss += iter_loss

                depth = hp_data.cpu().detach().numpy(
                ).copy()[0, 0, ...] * 1000
                depth_bgr = colorize_depth(depth, 100, 1500)
                depth_rgb = cv2.cvtColor(
                    depth_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)


                hanging_point_depth_gt \
                    = ground_truth[:, 1, ...].astype(np.float32) * 1000
                # rotations_gt = ground_truth[2:, ...]
                rotations_gt = ground_truth[0, 2:, ...]
                rotations_gt = rotations_gt.transpose(1, 2, 0)

                hanging_point_depth_gt_bgr \
                    = colorize_depth(hanging_point_depth_gt[0, ...], 100, 1500)
                    # = colorize_depth(hanging_point_depth_gt, 100, 1500)

                hanging_point_depth_gt_rgb = cv2.cvtColor(
                    hanging_point_depth_gt_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                rotations_mask = np.zeros((rotations_gt.shape[0],
                                           rotations_gt.shape[1]))
                for i in range(rotations_gt.shape[0]):
                    for j in range(rotations_gt.shape[1]):
                        if np.all(rotations_gt[i, j, :] != [1, 0, 0, 0]):
                            R = skrobot.coordinates.math.quaternion2matrix(
                                rotations_gt[i, j, :])
                            rotations_mask[i, j] = 255


                if index == test_data_num - 1:
                    print("Finish test {} data".format(index))
                    break

                if np.mod(index, 1) == 0:
                    print('epoch {}, {}/{},train loss is {}'.format(
                        epo,
                        index,
                        len(train_dataloader),
                        iter_loss))

                    vis.images(depth_rgb,
                               win='test depth_rgb',
                               opts=dict(
                                   title='test depth_rgb'))
                    # vis.images(depth_pred_rgb,
                    #            win='test depth_pred_rgb',
                    #            opts=dict(
                    #                title='test depth_pred_rgb'))

                    vis.images(hanging_point_depth_gt_rgb,
                               win='test hanging_point_depth_gt_rgb',
                               opts=dict(
                                   title='test hanging_point_depth_gt_rgb'))
                    # vis.images(rotations_mask,
                    #            win='test rotations_mask',
                    #            opts=dict(
                    #                title='test rotations_mask'))
                    # vis.images(confidence_mask.transpose(2, 0, 1),
                    # vis.images(confidence_mask,
                    vis.images(confidence_mask[0, ...],
                               win='test confidence_mask',
                               opts=dict(
                                   title='test confidence_mask'))
                    # vis.images([confidence_gt, confidence_np],
                    vis.images([confidence_gt[0, ...][None, ...],
                                confidence_np[0, ...][None, ...]],
                               win='train_confidence',
                               opts=dict(
                                   title='train confidence(GT, Pred)'))

        avg_test_loss = test_loss / len(test_dataloader)

        vis.line(X=np.array([epo]), Y=np.array([avg_train_loss]), win='loss',
                 name='avg_train_loss', update='append')
        vis.line(X=np.array([epo]), Y=np.array([avg_test_loss]), win='loss',
                 name='avg_test_loss', update='append')

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        if np.mod(epo, 1) == 0:
            torch.save(unet_model.state_dict(),
                       'checkpoints/unet_latestmodel_' + now + '.pt')
        print('epoch train loss = %f, epoch test loss = %f, best_loss = %f, %s'
              % (train_loss / len(train_dataloader),
                 test_loss / len(test_dataloader),
                 best_loss,
                 time_str))
        if best_loss > test_loss / len(test_dataloader):
            print('update best model {} -> {}'.format(
                best_loss, test_loss / len(test_dataloader)))
            best_loss = test_loss / len(test_dataloader)
            torch.save(unet_model.state_dict(),
                       'checkpoints/unet_bestmodel_' + now + '.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_path', '-dp', type=str,
                        help='Training data path',
                        default='/media/kosuke/SANDISK/meshdata/Hanging-ObjectNet3D-DoubleFaces/rotations_0514')
                        # default='/media/kosuke/SANDISK/meshdata/Hanging-ObjectNet3D-DoubleFaces/cup')
    parser.add_argument('--batch_size', '-bs', type=int,
                        help='max epoch',
                        default=32)
    parser.add_argument('--max_epoch', '-me', type=int,
                        help='max epoch',
                        default=1000000)
    parser.add_argument('--pretrained_model', '-p', type=str,
                        help='Pretrained model',
                        default='checkpoints/unet_latestmodel_20200514_0426.pt')

    parser.add_argument('--train_data_num', '-tr', type=int,
                        help='How much data to use for training',
                        default=1000000)
    parser.add_argument('--test_data_num', '-te', type=int,
                        help='How much data to use for testing',
                        default=1000000)

    args = parser.parse_args()
    train(data_path=args.data_path,
          batch_size=args.batch_size,
          max_epoch=args.max_epoch,
          pretrained_model=args.pretrained_model,
          train_data_num=args.train_data_num,
          test_data_num=args.test_data_num)
