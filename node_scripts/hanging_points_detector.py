#!/usr/bin/env python
# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import division

import copy
import os
import os.path as osp
import sys

import cameramodels
import message_filters
import numpy as np
import rospy
import torch
from geometry_msgs.msg import PoseArray, Pose
from torchvision import transforms
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from skrobot import coordinates
from skrobot.coordinates.math import quaternion2matrix
from hanging_points_cnn.learning_scripts.hpnet import HPNET
from hanging_points_cnn.utils.image import colorize_depth
from hanging_points_cnn.utils.image import draw_axis
from hanging_points_cnn.utils.image import normalize_depth
from hanging_points_cnn.utils.image import unnormalize_depth
from hanging_points_cnn.utils.image import remove_nan


try:
    import cv2
except ImportError:
    for path in sys.path:
        if '/opt/ros/' in path:
            print('sys.path.remove({})'.format(path))
            sys.path.remove(path)
            import cv2
            sys.path.append(path)


def frame_img(img, frame=1):
    height, width = img.shape[:2]
    resized_img = cv2.resize(img, (width - frame * 2, height - frame * 2))
    framed_img = np.zeros(img.shape)
    framed_img[frame:height - frame, frame:width - frame] = resized_img
    return framed_img


class HangingPointsNet():
    def __init__(self):
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("~output", Image, queue_size=10)
        self.pub_pred = rospy.Publisher("~output/pred", Image, queue_size=10)
        self.pub_depth = rospy.Publisher(
            "~colorized_depth", Image, queue_size=10)
        self.pub_axis = rospy.Publisher(
            "~axis", Image, queue_size=10)
        self.pub_axis_raw = rospy.Publisher(
            "~axis_raw", Image, queue_size=10)
        self.pub_hanging_points = rospy.Publisher(
            "/hanging_points", PoseArray, queue_size=10)

        self.gpu_id = rospy.get_param('~gpu', 0)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        pretrained_model = rospy.get_param(
            '~pretrained_model',
            # '/media/kosuke/SANDISK/hanging_points_net/checkpoints/resnet/hpnet_latestmodel_20200608_0311.pt')
            # '/media/kosuke/SANDISK/hanging_points_net/checkpoints/resnet/hpnet_latestmodel_20200619_2113.pt')
            '/media/kosuke/SANDISK/hanging_points_net/checkpoints/gray/hpnet_bestmodel_20200730_0301.pt')
        # '/media/kosuke/SANDISK/hanging_points_net/checkpoints/resnet/hpnet_bestmodel_20200527_2110.pt')
        # '/media/kosuke/SANDISK/hanging_points_net/checkpoints/resnet/hpnet_bestmodel_20200527_1846.pt')
        # '/media/kosuke/SANDISK/hanging_points_net/checkpoints/resnet/hpnet_bestmodel_20200527_0224.pt')

        # '../learning_scripts/checkpoints/unet_latestmodel_20200507_0438.pt')
        # '../learning_scripts/checkpoints/unet_latestmodel_20200506_2259.pt')

        self.transform = transforms.Compose([
            transforms.ToTensor()])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.config = {
            'output_channels': 1,
            'feature_extractor_name': 'resnet50',
            'confidence_thresh': 0.3,
            'use_bgr': True,
            'use_bgr2gray': True,
        }

        self.model = HPNET(self.config).to(device)
        if osp.exists(pretrained_model):
            print('use pretrained model')
            self.model.load_state_dict(torch.load(pretrained_model))
        self.model.eval()
        self.camera_model = None
        self.load_camera_info()
        self.use_coords = False
        # self.hanging_points_pose_array = PoseArray()
        self.subscribe()

    def subscribe(self):
        self.sub_camera_info = message_filters.Subscriber(
            '~camera_info', CameraInfo, queue_size=1, buff_size=2**24)
        self.sub_rgb_raw = message_filters.Subscriber(
            '~rgb_raw', Image, queue_size=1, buff_size=2**24)
        self.sub_rgb = message_filters.Subscriber(
            '~rgb', Image, queue_size=1, buff_size=2**24)
        self.sub_depth = message_filters.Subscriber(
            '~depth', Image, queue_size=1, buff_size=2**24)
        sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_camera_info,
             self.sub_rgb_raw,
             self.sub_rgb,
             self.sub_depth],
            queue_size=100,
            slop=0.1)
        sync.registerCallback(self.callback)

    def get_full_camera_info(self, camera_info):
        full_camera_info = copy.copy(camera_info)
        full_camera_info.roi.x_offset = 0
        full_camera_info.roi.y_offset = 0
        full_camera_info.roi.height = 0
        full_camera_info.roi.width = 0

        return full_camera_info

    def load_camera_info(self):
        print('load camera info')
        self.camera_info = rospy.wait_for_message(
            '~camera_info', CameraInfo)
        self.camera_model\
            = cameramodels.PinholeCameraModel.from_camera_info(
                self.camera_info)

    def callback(self, camera_info_msg, img_raw_msg, img_msg, depth_msg):
        ymin = camera_info_msg.roi.y_offset
        xmin = camera_info_msg.roi.x_offset
        ymax = camera_info_msg.roi.y_offset + camera_info_msg.roi.height
        xmax = camera_info_msg.roi.x_offset + camera_info_msg.roi.width
        self.camera_model.roi = [ymin, xmin, ymax, xmax]

        bgr_raw = self.bridge.imgmsg_to_cv2(img_raw_msg, "bgr8")
        bgr = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

        depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

        if depth is None or bgr is None:
            return

        remove_nan(depth)
        depth[depth < 200] = 0
        depth[depth > 1000] = 0

        # depth_bgr = colorize_depth(depth, 100, 1500)

        bgr = cv2.resize(bgr, (256, 256))
        # depth_bgr = cv2.resize(depth_bgr, (256, 256))

        # depth_bgr[np.where(np.all(cv2.resize(bgr, (256, 256)) == [0, 0, 0], axis=-1))] = [0, 0, 0]

        depth = cv2.resize(depth, (256, 256))

        depth = frame_img(depth)
        kernel = np.ones((10, 10), np.uint8)
        depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)
        depth_bgr = colorize_depth(depth, 300, 1000)
        # depth_bgr[np.where(np.all(bgr == [0, 0, 0], axis=-1))] = [0, 0, 0]
        depth_bgr[np.where(depth == 0)] = [0, 0, 0]
        in_feature = depth.copy().astype(np.float32) * 0.001

        if self.config['use_bgr2gray']:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (256, 256))[..., None] / 255.
            normalized_depth = normalize_depth(
                depth, 0.2, 1)[..., None]
            in_feature = np.concatenate(
                (normalized_depth, gray), axis=2).astype(np.float32)

        if self.transform:
            in_feature = self.transform(in_feature)

        in_feature = in_feature.to(self.device)
        in_feature = in_feature.unsqueeze(0)

        confidence, depth_and_rotation = self.model(in_feature)
        confidence = confidence[0, 0:1, ...]
        confidence_np = confidence.cpu().detach().numpy().copy() * 255
        confidence_np = confidence_np.transpose(1, 2, 0)
        confidence_np[confidence_np <= 0] = 0
        confidence_np[confidence_np >= 255] = 255
        confidence_img = confidence_np.astype(np.uint8)
        confidence_img = cv2.resize(confidence_img, (256, 256))

        axis_pred = bgr.copy()
        axis_pred_raw = bgr_raw.copy()

        dep_pred = []
        hanging_points_pose_array = PoseArray()
        for i, roi in enumerate(self.model.rois_list[0]):
            if roi.tolist() == [0, 0, 0, 0]:
                continue
            roi = roi.cpu().detach().numpy().copy()
            hanging_point_x = int((roi[0] + roi[2]) / 2)
            hanging_point_y = int((roi[1] + roi[3]) / 2)

            depth_roi_clip = depth[int(roi[1]):int(roi[3]),
                                   int(roi[0]):int(roi[2])]

            # dep_roi_clip = depth_roi_clip
            dep_roi_clip = depth_roi_clip[np.where(
                np.logical_and(depth_roi_clip > 200, depth_roi_clip < 1000))]

            depth_roi_clip_bgr = colorize_depth(depth_roi_clip, 200, 1000)
            # print(np.max(dep_roi_clip), np.mean(dep_roi_clip), np.median(dep_roi_clip), np.min(dep_roi_clip))
            dep_roi_clip = np.median(dep_roi_clip) * 0.001
            dep = depth_and_rotation[i, 0]
            # dep_pred.append(float(dep))

            if self.use_coords:
                q = depth_and_rotation[i, 1:].cpu().detach().numpy().copy()
                q /= np.linalg.norm(q)
            else:
                v = depth_and_rotation[i, 1:4].cpu().detach().numpy()
                v /= np.linalg.norm(v)
                coords = coordinates.Coordinates()
                coordinates.geo.orient_coords_to_axis(coords, v, 'x')
                q = coords.quaternion

            camera_model_crop_resize \
                = self.camera_model.crop_resize_camera_info(
                    target_size=[256, 256])

            hanging_point = np.array(
                camera_model_crop_resize.project_pixel_to_3d_ray(
                    [int(hanging_point_x),
                     int(hanging_point_y)]))

            length = float(dep_roi_clip) / \
                hanging_point[2]
            hanging_point *= length

            print(hanging_point, float(dep), dep_roi_clip)
            if dep_roi_clip == np.nan:
                continue

            hanging_point_pose = Pose()
            hanging_point_pose.position.x = hanging_point[0]
            hanging_point_pose.position.y = hanging_point[1]
            hanging_point_pose.position.z = hanging_point[2]
            hanging_point_pose.orientation.w = q[0]
            hanging_point_pose.orientation.x = q[1]
            hanging_point_pose.orientation.y = q[2]
            hanging_point_pose.orientation.z = q[3]
            hanging_points_pose_array.poses.append(hanging_point_pose)

            axis_pred_raw = cv2.rectangle(
                axis_pred_raw,
                (int(roi[0] * (xmax - xmin) / float(256) + xmin),
                 int(roi[1] * (ymax - ymin) / float(256) + ymin)),
                (int(roi[2] * (xmax - xmin) / float(256) + xmin),
                 int(roi[3] * (ymax - ymin) / float(256) + ymin)),
                (0, 255, 0), 1)
            axis_pred_raw = draw_axis(axis_pred_raw,
                                      quaternion2matrix(q),
                                      hanging_point,
                                      self.camera_model.full_K)

        axis_pred = self.camera_model.crop_image(
            axis_pred_raw, copy=True).astype(np.uint8)

        # draw pred rois
        # for roi in self.model.rois_list[0]:
        #     if roi.tolist() == [0, 0, 0, 0]:
        #         continue
        #     axis_pred = cv2.rectangle(
        #         axis_pred, (roi[0], roi[1]), (roi[2], roi[3]),
        #         (0, 255, 0), 3)

        pred_color = np.zeros_like(bgr, dtype=np.uint8)
        pred_color[..., 2] = confidence_img[..., 0]
        img2 = pred_color
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        mask_inv = cv2.bitwise_not(img2gray)
        fg = cv2.bitwise_or(img2, img2, mask=mask_inv)
        pred_color = cv2.bitwise_or(bgr, fg)

        msg_out = self.bridge.cv2_to_imgmsg(confidence_img, "mono8")
        msg_out.header.stamp = depth_msg.header.stamp

        pred_msg = self.bridge.cv2_to_imgmsg(pred_color, "bgr8")
        pred_msg.header.stamp = depth_msg.header.stamp

        colorized_depth_msg = self.bridge.cv2_to_imgmsg(depth_bgr, "bgr8")
        colorized_depth_msg.header.stamp = depth_msg.header.stamp

        axis_pred_msg = self.bridge.cv2_to_imgmsg(axis_pred, "bgr8")
        axis_pred_msg.header.stamp = depth_msg.header.stamp

        axis_pred_raw_msg = self.bridge.cv2_to_imgmsg(axis_pred_raw, "bgr8")
        axis_pred_raw_msg.header.stamp = depth_msg.header.stamp

        hanging_points_pose_array.header = camera_info_msg.header
        self.pub_pred.publish(pred_msg)
        self.pub.publish(msg_out)
        self.pub_depth.publish(colorized_depth_msg)
        self.pub_axis.publish(axis_pred_msg)
        self.pub_axis_raw.publish(axis_pred_raw_msg)
        self.pub_hanging_points.publish(hanging_points_pose_array)


def main(args):
    rospy.init_node("hanging_points_net", anonymous=False)
    hpn = HangingPointsNet()
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)
