#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import os
import os.path as osp
import sys
import warnings
import yaml
from datetime import datetime

import cameramodels
import numpy as np
import torch
import torch.optim as optim
import tqdm
import visdom
from skrobot.coordinates.math import rotation_matrix_from_axis
from skrobot.coordinates.math import quaternion2matrix

from hanging_points_cnn.learning_scripts.hpnet import HPNET
from hanging_points_cnn.learning_scripts.hpnet_loss import HPNETLoss
from hanging_points_cnn.learning_scripts.hanging_points_data \
    import load_dataset
from hanging_points_cnn.learning_scripts.hanging_points_data \
    import load_test_dataset
from hanging_points_cnn.utils.image import colorize_depth
from hanging_points_cnn.utils.image import draw_axis
from hanging_points_cnn.utils.image import draw_roi
from hanging_points_cnn.utils.image import unnormalize_depth
from hanging_points_cnn.utils.rois_tools import annotate_rois
from hanging_points_cnn.utils.rois_tools import get_value_gt
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

    def __init__(self, data_path, test_data_path,
                 batch_size, max_epoch, pretrained_model, train_data_num,
                 val_data_num, save_dir, lr, config=None,
                 train_depth=False, port=6006, object_list=None,
                 data_augmentation=True):

        if config is None:
            warnings.warn('confing is not specified. use defalut confing.')
            config = {
                'output_channels': 1,
                'feature_extractor_name': 'resnet50',
                'confidence_thresh': 0.3,
                'depth_range': [100, 1500],
                'use_bgr': True,
                'use_bgr2gray': True,
                'roi_padding': 50
            }
        self.config = config
        self.depth_range = config['depth_range']
        self.train_dataloader, self.val_dataloader\
            = load_dataset(data_path, batch_size,
                           use_bgr=self.config['use_bgr'],
                           use_bgr2gray=self.config['use_bgr2gray'],
                           depth_range=self.depth_range,
                           object_list=object_list,
                           data_augmentation=data_augmentation)
        self.test_dataloader \
            = load_test_dataset(test_data_path,
                                use_bgr=self.config['use_bgr'],
                                use_bgr2gray=self.config['use_bgr2gray'],
                                depth_range=self.depth_range)

        self.train_data_num = train_data_num
        self.val_data_num = val_data_num
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.lr = lr
        self.max_epoch = max_epoch
        self.time_now = datetime.now().strftime('%Y%m%d_%H%M')
        self.best_loss = 1e10

        self.camramodel = None
        self.target_size = [256, 256]

        self.vis = visdom.Visdom(port=port)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print('device is:{}'.format(self.device))

        self.model = HPNET(config).to(self.device)

        self.save_model_interval = 1
        if os.path.exists(pretrained_model):
            print('use pretrained model')
            self.model.load_state_dict(
                torch.load(pretrained_model), strict=False)

        self.prev_model = copy.deepcopy(self.model)

        self.best_loss = 1e10
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.999),
            eps=1e-10, weight_decay=0, amsgrad=False)
        self.prev_optimizer = copy.deepcopy(self.optimizer)

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epo: 0.9 ** epo)

        self.now = datetime.now().strftime('%Y%m%d_%H%M')
        self.use_coords = False
        self.train_depth = train_depth

    def step(self, dataloader, mode):
        print('Start {}'.format(mode))
        # self.model = self.prev_model
        if mode == 'train':
            self.model.train()
        elif mode == 'val' or mode == 'test':
            self.model.eval()

        loss_sum = 0
        confidence_loss_sum = 0
        depth_loss_sum = 0
        rotation_loss_sum = 0
        rotation_loss_count = 0

        for index, (hp_data, depth_image, camera_info_path, hp_data_gt, annotation_data) in tqdm.tqdm(
                enumerate(dataloader), total=len(dataloader),
                desc='{} epoch={}'.format(mode, self.epo), leave=False):

            # if index == 0:
            #     self.model = self.prev_model

            self.cameramodel\
                = cameramodels.PinholeCameraModel.from_yaml_file(
                    camera_info_path[0])
            self.cameramodel.target_size = self.target_size

            depth_image = hp_data.numpy().copy()[0, 0, ...]
            depth_image = np.nan_to_num(depth_image)
            depth_image = unnormalize_depth(
                depth_image, self.depth_range[0], self.depth_range[1])
            hp_data = hp_data.to(self.device)

            depth_image_bgr = colorize_depth(depth_image, ignore_value=self.depth_range[0])

            if mode == 'train':
                confidence, depth, rotation = self.model(hp_data)
            elif mode == 'val' or mode == 'test':
                with torch.no_grad():
                    confidence, depth, rotation = self.model(hp_data)

            confidence_np = confidence[0, ...].cpu(
            ).detach().numpy().copy()
            confidence_np[confidence_np >= 1] = 1.
            confidence_np[confidence_np <= 0] = 0.
            confidence_vis = cv2.cvtColor(confidence_np[0, ...] * 255,
                                          cv2.COLOR_GRAY2BGR)

            if mode != 'test':
                pos_weight = hp_data_gt.detach().numpy().copy()
                pos_weight = pos_weight[:, 0, ...]
                zeroidx = np.where(pos_weight < 0.5)
                nonzeroidx = np.where(pos_weight >= 0.5)
                pos_weight[zeroidx] = 0.5
                pos_weight[nonzeroidx] = 1.0
                pos_weight = torch.from_numpy(pos_weight)
                pos_weight = pos_weight.to(self.device)

                hp_data_gt = hp_data_gt.to(self.device)
                confidence_gt = hp_data_gt[:, 0:1, ...]
                rois_list_gt, rois_center_list_gt = find_rois(confidence_gt)

                criterion = HPNETLoss(self.use_coords).to(self.device)

                if self.model.rois_list is None or rois_list_gt is None:
                    return None, None

                annotated_rois = annotate_rois(
                    self.model.rois_list, rois_list_gt, annotation_data)

                confidence_loss, depth_loss, rotation_loss = criterion(
                    confidence, hp_data_gt, pos_weight,
                    depth, rotation, annotated_rois)

                if self.train_depth:
                    loss = confidence_loss + rotation_loss + depth_loss
                else:
                    loss = confidence_loss + rotation_loss

                if torch.isnan(loss):
                    print('loss is nan!!')
                    self.model = self.prev_model
                    self.optimizer = torch.optim.Adam(
                        self.model.parameters(), lr=self.lr,
                        betas=(0.9, 0.999), eps=1e-10, weight_decay=0,
                        amsgrad=False)
                    self.optimizer.load_state_dict(
                        self.prev_optimizer.state_dict())
                    continue
                else:
                    self.prev_model = copy.deepcopy(self.model)
                    self.prev_optimizer = copy.deepcopy(self.optimizer)

                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 5)
                    self.optimizer.step()

                axis_gt = depth_image_bgr.copy()

                confidence_gt_vis = cv2.cvtColor(confidence_gt[0, 0, ...].cpu(
                ).detach().numpy().copy() * 255, cv2.COLOR_GRAY2BGR)

                # Visualize gt axis and roi
                for roi, roi_c in zip(rois_list_gt[0], rois_center_list_gt[0]):
                    if roi.tolist() == [0, 0, 0, 0]:
                        continue
                    roi = roi.cpu().detach().numpy().copy()
                    cx = roi_c[0]
                    cy = roi_c[1]

                    depth_and_rotation_gt = get_value_gt(
                        [cx, cy], annotation_data[0])
                    rotation_gt = depth_and_rotation_gt[1:]
                    depth_gt_val = depth_and_rotation_gt[0]
                    unnormalized_depth_gt_val = unnormalize_depth(
                        depth_gt_val, self.depth_range[0], self.depth_range[1])

                    hanging_point_pose = np.array(
                        self.cameramodel.project_pixel_to_3d_ray(
                            [int(cx), int(cy)])) \
                        * unnormalized_depth_gt_val * 0.001

                    if self.use_coords:
                        rot = quaternion2matrix(rotation_gt),

                    else:
                        v = np.matmul(quaternion2matrix(rotation_gt),
                                      [1, 0, 0])
                        rot = rotation_matrix_from_axis(v, [0, 1, 0], 'xy')
                    try:
                        draw_axis(axis_gt,
                                  rot,
                                  hanging_point_pose,
                                  self.cameramodel.K)
                    except Exception:
                        print('Fail to draw axis')

                    confidence_gt_vis = draw_roi(
                        confidence_gt_vis, roi, val=depth_gt_val, gt=True)
                    axis_gt = draw_roi(axis_gt, roi, val=depth_gt_val, gt=True)

            # Visualize pred axis and roi
            axis_pred = depth_image_bgr.copy()

            for i, (roi, roi_c) in enumerate(
                    zip(self.model.rois_list[0],
                        self.model.rois_center_list[0])):

                if roi.tolist() == [0, 0, 0, 0]:
                    continue
                roi = roi.cpu().detach().numpy().copy()
                cx = roi_c[0]
                cy = roi_c[1]

                dep = depth[i].cpu().detach().numpy().copy()
                normalized_dep_pred = float(dep)
                dep = unnormalize_depth(
                    dep, self.depth_range[0], self.depth_range[1])

                confidence_vis = draw_roi(
                    confidence_vis, roi, val=normalized_dep_pred)
                axis_pred = draw_roi(axis_pred, roi, val=normalized_dep_pred)

                if mode != 'test':
                    if annotated_rois[i][2]:
                        confidence_vis = draw_roi(
                            confidence_vis, annotated_rois[i][0],
                            val=annotated_rois[i][1][0], gt=True)
                        axis_pred = draw_roi(
                            axis_pred, annotated_rois[i][0],
                            val=annotated_rois[i][1][0], gt=True)

                hanging_point_pose = np.array(
                    self.cameramodel.project_pixel_to_3d_ray(
                        [int(cx), int(cy)])) * float(dep * 0.001)

                if self.use_coords:
                    # have not check this yet
                    q = rotation[i].cpu().detach().numpy().copy()
                    q /= np.linalg.norm(q)
                    rot = quaternion2matrix(q)

                else:
                    v = rotation[i].cpu().detach().numpy()
                    v /= np.linalg.norm(v)
                    rot = rotation_matrix_from_axis(v, [0, 1, 0], 'xy')

                try:
                    draw_axis(axis_pred,
                              rot,
                              hanging_point_pose,
                              self.cameramodel.K)
                except Exception:
                    print('Fail to draw axis')

            axis_pred = cv2.cvtColor(
                axis_pred, cv2.COLOR_BGR2RGB)
            confidence_vis = cv2.cvtColor(
                confidence_vis, cv2.COLOR_BGR2RGB)

            if self.config['use_bgr']:
                if self.config['use_bgr2gray']:
                    in_gray = hp_data.cpu().detach().numpy().copy()[
                        0, 1:2, ...] * 255
                    in_gray = in_gray.transpose(1, 2, 0).astype(np.uint8)
                    in_gray = cv2.cvtColor(in_gray, cv2.COLOR_GRAY2RGB)
                    in_gray = in_gray.transpose(2, 0, 1)
                    in_img = in_gray
                else:
                    in_bgr = hp_data.cpu().detach().numpy().copy()[
                        0, 3:, ...].transpose(1, 2, 0)
                    in_rgb = cv2.cvtColor(
                        in_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                    in_img = in_rgb

            if mode != 'test':
                confidence_loss_sum += confidence_loss.item()

                axis_gt = cv2.cvtColor(
                    axis_gt, cv2.COLOR_BGR2RGB)
                confidence_gt_vis = cv2.cvtColor(
                    confidence_gt_vis, cv2.COLOR_BGR2RGB)

                if rotation_loss.item() > 0:
                    depth_loss_sum += depth_loss.item()
                    rotation_loss_sum += rotation_loss.item()
                    loss_sum = loss_sum \
                        + confidence_loss.item() \
                        + rotation_loss.item()
                    rotation_loss_count += 1

                if np.mod(index, 1) == 0:
                    print(
                        'epoch {}, {}/{},{} loss is confidence:{} rotation:{} depth:{}'.format(   # noqa
                            self.epo,
                            index,
                            len(dataloader),
                            mode,
                            confidence_loss.item(),
                            rotation_loss.item(),
                            depth_loss.item()))

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
                    self.vis.images([in_img],
                                    win='{} in_gray'.format(mode),
                                    opts=dict(
                        title='{} in_gray'.format(mode)))
            else:
                if self.config['use_bgr']:
                    self.vis.images(
                        [in_img, confidence_vis.transpose(2, 0, 1),
                         axis_pred.transpose(2, 0, 1)],
                        win='{}-{}'.format(
                            mode, index),
                        opts=dict(
                            title='{}-{} hanging_point_depth (pred)'.format(
                                mode, index)))
                else:
                    self.vis.images(
                        [confidence_vis.transpose(2, 0, 1),
                         axis_pred.transpose(2, 0, 1)],
                        win='{}-{}'.format(
                            mode, index),
                        opts=dict(
                            title='{}-{} hanging_point_depth (pred)'.format(
                                mode, index)))

            if np.mod(index, 1000) == 0:
                save_file = osp.join(
                    self.save_dir,
                    'hpnet_latestmodel_' + self.time_now + '.pt')
                print('save {}'.format(save_file))
                torch.save(
                    self.model.state_dict(), save_file,
                    _use_new_zipfile_serialization=False)

        if mode != 'test':
            if len(dataloader) > 0:
                avg_confidence_loss\
                    = confidence_loss_sum / len(dataloader)
                if rotation_loss_count > 0:
                    avg_rotation_loss\
                        = rotation_loss_sum / rotation_loss_count
                    avg_depth_loss\
                        = depth_loss_sum / rotation_loss_count
                    avg_loss\
                        = loss_sum / rotation_loss_count
                else:
                    avg_rotation_loss = 1e10
                    avg_depth_loss = 1e10
                    avg_loss = 1e10
            else:
                avg_loss = loss_sum
                avg_confidence_loss = confidence_loss_sum
                avg_rotation_loss = rotation_loss_sum
                avg_depth_loss = rotation_loss_sum

            self.vis.line(
                X=np.array([self.epo]), Y=np.array([avg_confidence_loss]),
                opts={'title': 'confidence'},
                win='confidence loss',
                name='{}_confidence_loss'.format(mode),
                update='append')
            if rotation_loss_count > 0:
                self.vis.line(
                    X=np.array([self.epo]), Y=np.array([avg_rotation_loss]),
                    opts={'title': 'rotation loss'},
                    win='rotation loss',
                    name='{}_rotation_loss'.format(mode),
                    update='append')
                self.vis.line(
                    X=np.array([self.epo]), Y=np.array([avg_depth_loss]),
                    opts={'title': 'depth loss'},
                    win='depth loss',
                    name='{}_depth_loss'.format(mode),
                    update='append')
                self.vis.line(
                    X=np.array([self.epo]), Y=np.array([avg_loss]),
                    opts={'title': 'loss'},
                    win='loss', name='{}_loss'.format(mode),
                    update='append')

            if mode == 'val':
                if np.mod(self.epo, self.save_model_interval) == 0:
                    save_file = osp.join(
                        self.save_dir,
                        'hpnet_latestmodel_' + self.time_now + '.pt')
                    print('save {}'.format(save_file))
                    torch.save(
                        self.model.state_dict(), save_file,
                        _use_new_zipfile_serialization=False)

                if self.best_loss > avg_loss:
                    print('update best model {} -> {}'.format(
                        self.best_loss, avg_loss))
                    self.best_loss = avg_loss
                    save_file = osp.join(
                        self.save_dir,
                        'hpnet_bestmodel_' + self.time_now + '.pt')
                    print('save {}'.format(save_file))
                    # For ros(python 2, torch 1.4)
                    torch.save(
                        self.model.state_dict(), save_file,
                        _use_new_zipfile_serialization=False)

    def train(self):
        for self.epo in range(self.max_epoch):
            self.step(self.train_dataloader, 'train')
            self.step(self.val_dataloader, 'val')
            self.step(self.test_dataloader, 'test')
            self.scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-g', '--gpu', type=int,
        help='gpu id. '
        'if this option does not work, '
        'run `CUDA_VISIBLE_DEVICES={gpu id} python train_hpnet.py`',
        default=0)
    parser.add_argument(
        '--data-path',
        '-dp',
        type=str,
        help='Training and Validation data path',
        # default='/media/kosuke/SANDISK-2/meshdata/hanging_object/0927')
        default='/media/kosuke55/SANDISK-2/meshdata/shapenet_pouring_render/1218_mug_cap_helmet_bowl')  # noqa
    parser.add_argument(
        '--test-data-path',
        '-tdp',
        type=str,
        help='Test data path',
        default='/home/kosuke55/catkin_ws/src/hanging_points_cnn/data/test_images_pouring')  # noqa
    parser.add_argument('--batch_size', '-bs', type=int,
                        help='batch size',
                        default=4)
    parser.add_argument('--max_epoch', '-me', type=int,
                        help='max epoch',
                        default=1000000)
    parser.add_argument(
        '--pretrained_model',
        '-p',
        type=str,
        help='Pretrained model',
        # default='/media/kosuke/SANDISK/hanging_points_net/checkpoints/gray/hpnet_latestmodel_20201014_0347.pt')  # noqa
        default='/media/kosuke55/SANDISK-2/meshdata/shapenet_pouring_render/1218_mug_cap_helmet_bowl/hpnet_latestmodel_20201218_1032_fix.pt')  # noqa
    parser.add_argument(
        '--train_data_num', '-tr', type=int,
        help='How much data to use for training',
        default=1000000)
    parser.add_argument(
        '--val_data_num', '-te', type=int,
        help='How much data to use for validation',
        default=1000000)
    parser.add_argument(
        '--confing', '-c', type=str,
        help='config',
        default='config/gray_model.yaml')
    parser.add_argument(
        '--port', type=int,
        help='port',
        default=6006)
    parser.add_argument(
        '--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument(
        '--train-depth', '-td',
        action='store_true',
        help='if true, train depth')
    parser.add_argument(
        '--disable_data_augmentation', '-dda',
        action='store_false',
        help='if true, disable_data_augmentation')
    parser.add_argument(
        '--object_list',
        '-ol',
        type=str,
        help='list of objects used for traing',
        default=None)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    data_augmentation = args.disable_data_augmentation

    with open(args.confing) as f:
        config = yaml.safe_load(f)

    object_list_file = args.object_list
    if object_list_file is None:
        object_list = None
    else:
        if osp.isfile(object_list_file):
            with open(object_list_file) as f:
                object_list = [s.strip() for s in f.readlines()]

    trainer = Trainer(
        data_path=args.data_path,
        test_data_path=args.test_data_path,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        pretrained_model=args.pretrained_model,
        train_data_num=args.train_data_num,
        val_data_num=args.val_data_num,
        save_dir=args.data_path,
        lr=args.lr,
        config=config,
        train_depth=args.train_depth,
        port=args.port,
        object_list=object_list,
        data_augmentation=data_augmentation)
    trainer.train()
