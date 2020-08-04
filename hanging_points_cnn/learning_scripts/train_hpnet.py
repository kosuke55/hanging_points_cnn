#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import os.path as osp
import sys
import yaml
from datetime import datetime
from pathlib import Path

import cameramodels
import numpy as np
import torch
import torch.optim as optim
import tqdm
import visdom
from skrobot.coordinates.math import quaternion2matrix

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))  # noqa:
from hpnet import HPNET
from HPNETLoss import HPNETLoss
from HangingPointsData import load_dataset
from hanging_points_cnn.utils.image import colorize_depth
from hanging_points_cnn.utils.image import create_depth_circle
from hanging_points_cnn.utils.image import draw_axis
from hanging_points_cnn.utils.image import unnormalize_depth
from hanging_points_cnn.utils.rois_tools import annotate_rois
from hanging_points_cnn.utils.rois_tools import find_rois

try:
    import cv2
except ImportError:
    for path in sys.path:
        if '/opt/ros/' in path:
            print('sys.path.remove({})'.format(path))
            sys.path.remove(path)
            import cv2
            sys.path.append(path)


class Trainer(object):

    def __init__(self, gpu, data_path, batch_size, max_epoch,
                 pretrained_model, train_data_num, val_data_num,
                 save_dir, lr, config=None, port=6006):

        if config is None:
            config = {
                'output_channels': 1,
                'feature_extractor_name': 'resnet50',
                'confidence_thresh': 0.3,
                'use_bgr': True,
                'use_bgr2gray': True,
            }
        self.config = config
        self.depth_range = [0.2, 0.7]
        self.train_dataloader, self.val_dataloader = load_dataset(
            data_path, batch_size,
            use_bgr=self.config['use_bgr'],
            use_bgr2gray=self.config['use_bgr2gray'],
            depth_range=self.depth_range)

        self.train_data_num = train_data_num
        self.val_data_num = val_data_num
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.max_epoch = max_epoch
        self.time_now = datetime.now().strftime('%Y%m%d_%H%M')
        self.best_loss = 1e10

        self.intrinsics = np.load(
            sorted(list(Path(data_path).glob("**/intrinsics.npy")))[0])

        # for visualize
        self.height = 1080
        self.width = 1920
        self.cameramodel = cameramodels.PinholeCameraModel.from_intrinsic_matrix(
            self.intrinsics, self.height, self.width)

        self.vis = visdom.Visdom(port=port)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print('device is:{}'.format(self.device))

        self.model = HPNET(config).to(self.device)
        self.save_model_interval = 1
        if os.path.exists(pretrained_model):
            print('use pretrained model')
            self.model.load_state_dict(
                torch.load(pretrained_model), strict=False)

        self.best_loss = 1e10
        # self.optimizer = optim.SGD(
        #     self.model.parameters(), lr=args.lr, momentum=0.5, weight_decay=1e-6)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr,  betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epo: 0.9 ** epo)

        self.now = datetime.now().strftime('%Y%m%d_%H%M')

    def step(self, dataloader, mode):
        print('Start {}'.format(mode))

        if mode == 'train':
            self.model.train()
        elif mode == 'val':
            self.model.eval()

        loss_sum = 0
        confidence_loss_sum = 0
        depth_loss_sum = 0
        rotation_loss_sum = 0
        rotation_loss_count = 0

        for index, (hp_data, depth, clip_info, hp_data_gt) in tqdm.tqdm(
                enumerate(dataloader), total=len(dataloader),
                desc='{} epoch={}'.format(mode, self.epo), leave=False):
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
            pos_weight = pos_weight.to(self.device)

            criterion = HPNETLoss().to(self.device)
            hp_data = hp_data.to(self.device)

            depth = depth.numpy(
            ).copy()[0, 0, ...] * 1000
            depth_bgr = colorize_depth(depth.copy(), 100, 1500)

            hp_data_gt = hp_data_gt.to(self.device)
            ground_truth = hp_data_gt.cpu().detach().numpy().copy()
            confidence_gt = hp_data_gt[:, 0:1, ...]
            rois_list_gt, rois_center_list_gt = find_rois(confidence_gt)

            confidence, depth_and_rotation = self.model(hp_data)
            confidence_np = confidence[0, ...].cpu(
            ).detach().numpy().copy()
            confidence_np[confidence_np >= 1] = 1.
            confidence_np[confidence_np <= 0] = 0.
            confidence_vis = cv2.cvtColor(confidence_np[0, ...] * 255,
                                          cv2.COLOR_GRAY2BGR)

            depth_and_rotation_gt = hp_data_gt[:, 1:, ...]
            if self.model.rois_list is None or rois_list_gt is None:
                return None, None
            annotated_rois = annotate_rois(
                self.model.rois_list, rois_list_gt, depth_and_rotation_gt)

            confidence_loss, depth_loss, rotation_loss = criterion(
                confidence, hp_data_gt, pos_weight, depth_and_rotation, annotated_rois)

            loss = confidence_loss * 0.1 + rotation_loss

            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # if self.config['use_bgr']:
            #     depth_bgr = hp_data.cpu().detach().numpy()[
            #         0, 0:3, ...].transpose(1, 2, 0)
            # else:
            #     depth = hp_data.cpu().detach().numpy(
            #     ).copy()[0, 0, ...] * 1000
            #     depth_bgr = colorize_depth(depth.copy(), 100, 1500)

            # hanging_point_depth_gt \
            #     = ground_truth[0, 1, ...].astype(np.float32) * 1000
            hanging_point_depth_gt \
                = unnormalize_depth(
                    ground_truth[0, 1, ...].astype(np.float32),
                    self.depth_range[0], self.depth_range[1])

            rotations_gt = ground_truth[0, 2:, ...]i
            rotations_gt = rotations_gt.transpose(1, 2, 0)
            hanging_point_depth_gt_bgr \
                = colorize_depth(hanging_point_depth_gt, 100, 1500)
            hanging_point_depth_gt_rgb = cv2.cvtColor(
                hanging_point_depth_gt_bgr,
                cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

            axis_gt = depth_bgr.copy()
            axis_large_gt = np.zeros((self.height, self.width, 3))
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
                cx = roi_c[0]
                cy = roi_c[1]
                dep = depth[int(roi[1]):int(roi[3]),
                            int(roi[0]):int(roi[2])]
                dep = np.median(dep[np.where(
                    np.logical_and(dep > 200, dep < 1000))]).astype(np.uint8)
                dep_gt.append(dep)
                rotation = rotations_gt[cy, cx, :]
                pixel_point = [int(cx * (xmax - xmin) / float(256) + xmin),
                               int(cy * (ymax - ymin) / float(256) + ymin)]
                hanging_point_pose = np.array(
                    self.cameramodel.project_pixel_to_3d_ray(
                        pixel_point)) * dep * 0.001
                try:
                    draw_axis(axis_large_gt,
                              quaternion2matrix(rotation),
                              hanging_point_pose,
                              self.intrinsics)
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
            axis_large_pred = np.zeros((self.height, self.width, 3))
            axis_large_pred[ymin:ymax, xmin:xmax]\
                = cv2.resize(axis_pred, (xmax - xmin, ymax - ymin))
            dep_pred = []
            # for i, roi in enumerate(self.model.rois_list[0]):
            for i, (roi, roi_c) in enumerate(zip(self.model.rois_list[0], self.model.rois_center_list[0])):

                if roi.tolist() == [0, 0, 0, 0]:
                    continue
                roi = roi.cpu().detach().numpy().copy()
                cx = roi_c[0]
                cy = roi_c[1]

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

                create_depth_circle(depth, cy, cx, dep)

                q = depth_and_rotation[i, 1:].cpu().detach().numpy().copy()
                q /= np.linalg.norm(q)
                pixel_point = [int(cx * (xmax - xmin) / float(256) + xmin),
                               int(cy * (ymax - ymin) / float(256) + ymin)]
                hanging_point_pose = np.array(
                    self.cameramodel.project_pixel_to_3d_ray(
                        pixel_point)) * float(dep * 0.001)
                try:
                    draw_axis(axis_large_pred,
                              quaternion2matrix(q),
                              hanging_point_pose,
                              self.intrinsics)
                except Exception:
                    print('Fail to draw axis')
                    pass

            axis_pred = cv2.resize(axis_large_pred[ymin:ymax, xmin:xmax],
                                   (256, 256)).astype(np.uint8)
            axis_pred = cv2.cvtColor(
                axis_pred, cv2.COLOR_BGR2RGB)

            depth_pred_bgr = colorize_depth(depth, 100, 1500)
            depth_pred_rgb = cv2.cvtColor(
                depth_pred_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

            # draw pred rois
            for i, roi in enumerate(self.model.rois_list[0]):
                if roi.tolist() == [0, 0, 0, 0]:
                    continue
                axis_pred = cv2.rectangle(
                    axis_pred, (roi[0], roi[1]), (roi[2], roi[3]),
                    (0, 255, 0), 3)
                if annotated_rois[i][2]:
                    axis_pred = cv2.rectangle(
                        axis_pred, (int(annotated_rois[i][0][0]), int(annotated_rois[i][0][1])), (
                            int(annotated_rois[i][0][2]), int(annotated_rois[i][0][3])), (255, 0, 0), 2)
            confidence_loss_sum += confidence_loss.item()

            if rotation_loss.item() > 0:
                rotation_loss_sum += rotation_loss.item()
                loss_sum = loss_sum + confidence_loss.item() + rotation_loss.item()
                rotation_loss_count += 1

            if np.mod(index, 1) == 0:
                print('epoch {}, {}/{},{} loss is confidence:{} rotation:{}'.format(
                    self.epo,
                    index,
                    len(dataloader),
                    mode,
                    confidence_loss_sum,
                    rotation_loss_sum
                ))

                self.vis.images([hanging_point_depth_gt_rgb,
                                 depth_pred_rgb],
                                win='{} hanging_point_depth_gt_rgb'.format(
                                    mode),
                                opts=dict(
                    title='{} hanging_point_depth (GT, pred)'.format(mode)))
                self.vis.images([axis_gt.transpose(2, 0, 1),
                                 axis_pred.transpose(2, 0, 1)],
                                win='{} axis'.format(mode),
                                opts=dict(
                    title='{} axis'.format(mode)))
                self.vis.images([confidence_gt_vis.transpose(2, 0, 1),
                                 confidence_vis.transpose(2, 0, 1)],
                                win='{}_confidence_roi'.format(mode),
                                opts=dict(
                    title='{} confidence(GT, Pred)'.format(mode)))

                if self.config['use_bgr']:
                    if self.config['use_bgr2gray']:
                        in_gray = hp_data.cpu().detach().numpy().copy()[
                            0, 1:2, ...] * 255
                        in_gray = in_gray.astype(np.uint8)
                        print(in_gray.shape)
                        self.vis.images([in_gray],
                                        win='{} in_gray'.format(mode),
                                        opts=dict(
                            title='{} in_gray'.format(mode)))
                    else:
                        in_bgr = hp_data.cpu().detach().numpy().copy()[
                            0, 3:, ...].transpose(1, 2, 0)
                        in_rgb = cv2.cvtColor(
                            in_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                        self.vis.images([in_rgb],
                                        win='{} in_rgb'.format(mode),
                                        opts=dict(
                            title='{} in_rgb'.format(mode)))

        if len(dataloader) > 0:
            avg_confidence_loss\
                = confidence_loss_sum / len(dataloader)
            if rotation_loss_count > 0:
                avg_rotation_loss\
                    = rotation_loss_sum / rotation_loss_count
                avg_loss\
                    = loss_sum / rotation_loss_count
            else:
                avg_rotation_loss = 1e10
                avg_loss = 1e10
        else:
            avg_confidence_loss = confidence_loss_sum
            avg_rotation_loss = rotation_loss_sum

        self.vis.line(X=np.array([self.epo]), Y=np.array([avg_confidence_loss]),
                      win='loss', name='{}_confidence_loss'.format(mode), update='append')
        if rotation_loss_count > 0:
            self.vis.line(X=np.array([self.epo]), Y=np.array([avg_rotation_loss]),
                          win='loss', name='{}_rotation_loss'.format(mode), update='append')
            self.vis.line(X=np.array([self.epo]), Y=np.array([avg_loss]),
                          win='loss', name='{}_loss'.format(mode), update='append')

        if mode == 'val':
            if np.mod(self.epo, self.save_model_interval) == 0:
                torch.save(
                    self.model.state_dict(),
                    osp.join(self.save_dir, 'hpnet_latestmodel_' + self.time_now + '.pt'))
            if self.best_loss > avg_loss:
                print('update best model {} -> {}'.format(
                    self.best_loss, avg_loss))
                self.best_loss = avg_loss
                torch.save(
                    self.model.state_dict(),
                    osp.join(self.save_dir, 'hpnet_bestmodel_' + self.time_now + '.pt'))

    def train(self):
        for self.epo in range(self.max_epoch):
            self.step(self.train_dataloader, 'train')
            # if np.mod(self.epo, 50) == 0:
            self.step(self.val_dataloader, 'val')
            self.scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument(
        '--data_path',
        '-dp',
        type=str,
        help='Training data path',
        # default='/media/kosuke/SANDISK/meshdata/ycb_hanging_object/runmany')
        # default='/media/kosuke/SANDISK/meshdata/ycb_hanging_object/0722-only2')
        default='/media/kosuke/SANDISK/meshdata/ycb_hanging_object/0722')
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
        default='/media/kosuke/SANDISK/hanging_points_net/checkpoints/hoge/hpnet_latestmodel_20200727_1438.pt')
    parser.add_argument('--train_data_num', '-tr', type=int,
                        help='How much data to use for training',
                        default=1000000)
    parser.add_argument('--val_data_num', '-te', type=int,
                        help='How much data to use for validation',
                        default=1000000)
    parser.add_argument('--confing', '-c', type=str,
                        help='config',
                        default='config/gray_model.yaml')
    parser.add_argument('--port', type=int,
                        help='port',
                        default=6006)

    parser.add_argument(
        '--save_dir',
        '-sd',
        type=str,
        help='pt model save dir',
        default='/media/kosuke/SANDISK/hanging_points_net/checkpoints/gray')
    # default='/media/kosuke/SANDISK/hanging_points_net/checkpoints/resnet')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')

    args = parser.parse_args()

    with open(args.confing) as f:
        config = yaml.safe_load(f)

    trainer = Trainer(
        gpu=args.gpu,
        data_path=args.data_path,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        pretrained_model=args.pretrained_model,
        train_data_num=args.train_data_num,
        val_data_num=args.val_data_num,
        save_dir=args.save_dir,
        lr=args.lr,
        config=config,
        port=args.port)
    trainer.train()
