#!/usr/bin/env python
# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import division

import os.path as osp
import sys

import cameramodels
import message_filters
import numpy as np
import rospy
import torch
from torchvision import transforms
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from skrobot.coordinates.math import quaternion2matrix

from learning_scripts.hpnet import HPNET
from utils.visualize import colorize_depth, draw_axis

try:
    import cv2
except ImportError:
    for path in sys.path:
        if '/opt/ros/' in path:
            print('sys.path.remove({})'.format(path))
            sys.path.remove(path)
            import cv2
            sys.path.append(path)

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
        self.camera_info = rospy.get_param(
            '~camera_info',
            '/apply_mask_image/output/camera_info')
        self.input_image_raw = rospy.get_param(
            '~image',
            'camera/color/image_rect_color')
        self.input_image = rospy.get_param(
            '~image',
            '/apply_mask_image/output')
        self.input_depth = rospy.get_param(
            '~image',
            '/apply_mask_depth/output')
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_model = rospy.get_param(
            '~pretrained_model',
            '/media/kosuke/SANDISK/hanging_points_net/checkpoints/resnet/hpnet_latestmodel_20200522_0258.pt')
        # '../learning_scripts/checkpoints/unet_latestmodel_20200507_0438.pt')
        # '../learning_scripts/checkpoints/unet_latestmodel_20200506_2259.pt')

        self.transform = transforms.Compose([
            transforms.ToTensor()])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = {
            'feature_compress': 1 / 16,
            'num_class': 6,
            'pool_out_size': 8,
            'confidence_thresh': 0.3,
        }
        self.model = HPNET(config).to(device)
        if osp.exists(pretrained_model):
            print('use pretrained model')
            self.model.load_state_dict(torch.load(pretrained_model))
        self.model.eval()
        self.camera_model = None
        self.load_camera_info()
        self.subscribe()

    def subscribe(self):
        self.sub_camera_info = message_filters.Subscriber(
            self.camera_info, CameraInfo, queue_size=1, buff_size=2**24)
        self.sub_rgb_raw = message_filters.Subscriber(
            self.input_image_raw, Image, queue_size=1, buff_size=2**24)
        self.sub_rgb = message_filters.Subscriber(
            self.input_image, Image, queue_size=1, buff_size=2**24)
        self.sub_depth = message_filters.Subscriber(
            self.input_depth, Image, queue_size=1, buff_size=2**24)
        sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_camera_info,
             self.sub_rgb_raw,
             self.sub_rgb,
             self.sub_depth],
            queue_size=100,
            slop=0.1)
        sync.registerCallback(self.callback)

    # def colorize_depth(self, depth, min_value=None, max_value=None):
    #     min_value = np.nanmin(depth) if min_value is None else min_value
    #     max_value = np.nanmax(depth) if max_value is None else max_value

    #     gray_depth = depth.copy()
    #     nan_mask = np.isnan(gray_depth)
    #     gray_depth[nan_mask] = 0
    #     gray_depth = 255 * (gray_depth - min_value) / (max_value - min_value)
    #     gray_depth[gray_depth < 0] = 0
    #     gray_depth[gray_depth > 255] = 255
    #     gray_depth = gray_depth.astype(np.uint8)
    #     colorized = cv2.applyColorMap(gray_depth, cv2.COLORMAP_JET)
    #     colorized[nan_mask] = (0, 0, 0)

    #     return colorized

    def load_camera_info(self):
        print('load camera info')
        camera_info = rospy.wait_for_message(self.camera_info, CameraInfo)
        print(camera_info)
        self.camera_model \
            = cameramodels.PinholeCameraModel.from_camera_info(
                camera_info)

    def callback(self, camera_info_msg, img_raw_msg, img_msg, depth_msg):
        xmin = camera_info_msg.roi.x_offset
        xmax = camera_info_msg.roi.x_offset + camera_info_msg.roi.width
        ymin = camera_info_msg.roi.y_offset
        ymax = camera_info_msg.roi.y_offset + camera_info_msg.roi.height

        bgr_raw = self.bridge.imgmsg_to_cv2(img_raw_msg, "bgr8")
        bgr = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

        if depth is None or bgr is None:
            return

        # print("depth min", np.min(depth))
        # print("depth mean", np.mean(depth))
        # print("depth max", np.max(depth))
        depth_bgr = colorize_depth(depth, 100, 1500)

        bgr = cv2.resize(bgr, (256, 256))
        depth_bgr = cv2.resize(depth_bgr, (256, 256))
        depth_bgr[np.where(np.all(bgr == [0, 0, 0], axis=-1))] = [0, 0, 0]
        # depth_bgr[np.where(np.all(cv2.resize(bgr, (256, 256)) == [0, 0, 0], axis=-1))] = [0, 0, 0]

        depth = cv2.resize(depth, (256, 256))
        in_feature = depth.copy().astype(np.float32) * 0.001

        # print('in_feature.shape)', in_feature.shape)

        if self.transform:
            in_feature = self.transform(in_feature)

        in_feature = in_feature.to(self.device)
        in_feature = in_feature.unsqueeze(0)

        confidence, depth_and_rotation = self.model(in_feature)
        confidence = confidence[0, 0:1, ...]
        confidence_np = confidence.cpu().detach().numpy().copy() * 255
        confidence_np = confidence_np.transpose(1, 2, 0) * 2
        confidence_np[confidence_np <= 0] = 0
        confidence_np[confidence_np >= 255] = 255
        confidence_img = confidence_np.astype(np.uint8)
        confidence_img = cv2.resize(confidence_img, (256, 256))

        axis_pred = bgr.copy()
        axis_pred_raw = bgr_raw.copy()
        # axis_large_pred = np.zeros((self.camera_model.height,
        #                             self.camera_model.width, 3))

        # axis_large_pred[ymin:ymax, xmin:xmax] \
        #     = cv2.resize(axis_pred, (xmax - xmin, ymax - ymin))

        dep_pred = []
        for i, roi in enumerate(self.model.rois_list[0]):
            if roi.tolist() == [0, 0, 0, 0]:
                continue
            roi = roi.cpu().detach().numpy().copy()
            cx = int((roi[0] + roi[2]) / 2)
            cy = int((roi[1] + roi[3]) / 2)
            dep = depth_and_rotation[i, 0] * 1000
            dep_pred.append(float(dep))

            q = depth_and_rotation[i, 1:].cpu().detach().numpy().copy()
            q /= np.linalg.norm(q)
            pixel_point = [int(cx * (xmax - xmin) / float(256) + xmin),
                           int(cy * (ymax - ymin) / float(256) + ymin)]
            hanging_point_pose = np.array(
                self.camera_model.project_pixel_to_3d_ray(
                    pixel_point)) * float(dep * 0.001)
            axis_pred_raw = cv2.rectangle(
                axis_pred_raw,
                (int(roi[0] * (xmax - xmin) / float(256) + xmin),
                 int(roi[1] * (ymax - ymin) / float(256) + ymin)),
                (int(roi[2] * (xmax - xmin) / float(256) + xmin),
                 int(roi[3] * (ymax - ymin) / float(256) + ymin)),
                (0, 255, 0), 1)
            draw_axis(axis_pred_raw,
                      quaternion2matrix(q),
                      hanging_point_pose,
                      self.camera_model.K)

        # print('dep_pred', dep_pred)

        axis_pred = cv2.resize(axis_pred_raw[ymin:ymax, xmin:xmax],
                               (256, 256)).astype(np.uint8)

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
        msg_out.header.stamp = img_msg.header.stamp

        pred_msg = self.bridge.cv2_to_imgmsg(pred_color, "bgr8")
        pred_msg.header.stamp = img_msg.header.stamp

        colorized_depth_msg = self.bridge.cv2_to_imgmsg(depth_bgr, "bgr8")
        colorized_depth_msg.header.stamp = img_msg.header.stamp

        axis_pred_msg = self.bridge.cv2_to_imgmsg(axis_pred, "bgr8")
        axis_pred_msg.header.stamp = img_msg.header.stamp

        axis_pred_raw_msg = self.bridge.cv2_to_imgmsg(axis_pred_raw, "bgr8")
        axis_pred_raw_msg.header.stamp = img_msg.header.stamp

        self.pub_pred.publish(pred_msg)
        self.pub.publish(msg_out)
        self.pub_depth.publish(colorized_depth_msg)
        self.pub_axis.publish(axis_pred_msg)
        self.pub_axis_raw.publish(axis_pred_raw_msg)


def main(args):
    rospy.init_node("hanging_points_net", anonymous=False)
    hpn = HangingPointsNet()
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)
