#!/usr/bin/env python
# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import division

import os
import os.path as osp
import sys

import cv2
import image_geometry
import message_filters
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import torch
import torch.optim as optim
import visdom
from cv_bridge import CvBridge
from skimage.morphology import convex_hull_image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image, PointCloud2

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from learning_scripts.torchvision import transforms
from learning_scripts.hpnet import HPNET
from learning_scripts.HPNETLoss import HPNETLoss
# from learning_scripts.UNET import UNET


class HangingPointsNet():
    def __init__(self):
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("~output", Image, queue_size=10)
        self.pub_pred = rospy.Publisher("~output/pred", Image, queue_size=10)
        self.pub_depth = rospy.Publisher(
            "~colorized_depth", Image, queue_size=10)
        self.input_image_raw = rospy.get_param(
            '~image',
            'camera/color/image_rect_color')
        self.input_image = rospy.get_param(
            '~image',
            '/apply_mask_image/output')
        self.camera_info = rospy.get_param(
            '~camera_info',
            '/apply_mask_image/output/camera_info')
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

        # pretrained_model = rospy.get_param(
        #     '~pretrained_model',
        #     'checkpoints/mango_best.pt')
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        # self.model = BCNN()
        # self.model = BCNN().to(self.device)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = UNET(in_channels=3).to(device)
        self.model = HPNET().to(device)
        if os.path.exists(pretrained_model):
            print('use pretrained model')
            self.model.load_state_dict(torch.load(pretrained_model))
        self.model.eval()
        self.subscribe()

    def subscribe(self):
        self.sub_rgb_raw = message_filters.Subscriber(
            self.input_image_raw, Image, queue_size=1, buff_size=2**24
        )
        self.sub_rgb = message_filters.Subscriber(
            self.input_image, Image, queue_size=1, buff_size=2**24
        )
        self.sub_depth = message_filters.Subscriber(
            self.input_depth, Image, queue_size=1, buff_size=2**24
        )
        sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb_raw,
             self.sub_rgb,
             self.sub_depth,
             self.camera_info],
            queue_size=100,
            slop=0.1,
        )
        sync.registerCallback(self.callback)
        # self.image_sub = rospy.Subscriber(
        #     self.input_image, Image, self.callback, )

    def colorize_depth(self, depth, min_value=None, max_value=None):
        min_value = np.nanmin(depth) if min_value is None else min_value
        max_value = np.nanmax(depth) if max_value is None else max_value

        gray_depth = depth.copy()
        nan_mask = np.isnan(gray_depth)
        gray_depth[nan_mask] = 0
        gray_depth = 255 * (gray_depth - min_value) / (max_value - min_value)
        gray_depth[gray_depth < 0] = 0
        gray_depth[gray_depth > 255] = 255
        gray_depth = gray_depth.astype(np.uint8)
        colorized = cv2.applyColorMap(gray_depth, cv2.COLORMAP_JET)
        colorized[nan_mask] = (0, 0, 0)

        return colorized

    def callback(self, img_raw_msg, img_msg, depth_msg, ):
        # bgr = self.bridge.imgmsg_to_cv2(img_msg, "rgb8")
        bgr = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

        if depth is None or bgr is None:
            return

        print("depth min", np.min(depth))
        print("depth mean", np.mean(depth))
        print("depth max", np.max(depth))
        depth_bgr = self.colorize_depth(depth, 100, 1500)
        depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)

        bgr = cv2.resize(bgr, (256, 256))
        depth_bgr = cv2.resize(depth_bgr, (256, 256))
        depth_bgr[np.where(np.all(bgr == [0, 0, 0], axis=-1))] = [0, 0, 0]

        in_feature = depth.copy() * 0.001

        if self.transform:
            in_feature = self.transform(in_feature)

        in_feature = in_feature.to(self.device)
        in_feature = in_feature.unsqueeze(0)

        # output = self.model(in_feature)
        confidence, depth_and_rotation = self.model(in_feature)
        confidence = confidence[0, 0:1, ...]
        confidence_np = confidence.cpu().detach().numpy().copy()
        confidence_np = confidence_np.transpose(1, 2, 0) * 2
        confidence_np[confidence_np <= 0] = 0
        confidence_np[confidence_np >= 255] = 255
        confidence_img = confidence_np.astype(np.uint8)


        # -----
        # Visualize pred axis and roi
        axis_pred = depth_bgr.copy()
        axis_large_pred = np.zeros((1080, 1920, 3))
        axis_large_pred[ymin:ymax, xmin:xmax] \
            = cv2.resize(axis_pred, (xmax - xmin, ymax - ymin))
        dep_pred = []
        for i, roi in enumerate(self.model.rois_list[0]):
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

            q = depth_and_rotation[i, 1:].cpu().detach().numpy().copy()
            q /= np.linalg.norm(q)
            pixel_point = [int(cx * (xmax - xmin) / float(256) + xmin),
                           int(cy * (ymax - ymin) / float(256) + ymin)]
            hanging_point_pose = np.array(
                cameramodel.project_pixel_to_3d_ray(
                    pixel_point)) * float(dep * 0.001)
            # try:
            draw_axis(axis_large_pred,
                      skrobot.coordinates.math.quaternion2matrix(
                          q),
                      hanging_point_pose,
                      intrinsics)
            # except Exception:
                # print('Fail to draw axis')
                # pass

        print('dep_pred', dep_pred)

        axis_pred = cv2.resize(axis_large_pred[ymin:ymax, xmin:xmax],
                               (256, 256)).astype(np.uint8)
        axis_pred = cv2.cvtColor(
            axis_pred, cv2.COLOR_BGR2RGB)

        depth_pred_bgr = colorize_depth(depth, 100, 1500)
        depth_pred_rgb = cv2.cvtColor(
            depth_pred_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

        # draw pred rois
        for roi in self.model.rois_list[0]:
            if roi.tolist() == [0, 0, 0, 0]:
                continue
            axis_pred = cv2.rectangle(
                axis_pred, (roi[0], roi[1]), (roi[2], roi[3]),
                (0, 255, 0), 3)
        # -----


        # annotation_img = annotation_img.astype(np.uint8)
        pred_color = np.zeros_like(bgr, dtype=np.uint8)
        pred_color[..., 2] = confidence_img[..., 0]

        img2 = pred_color
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        mask_inv = cv2.bitwise_not(img2gray)
        # white_background = np.full(img2.shape, 255, dtype=np.uint8)
        # bk = cv2.bitwise_or(white_background, white_background, mask=mask_inv)
        fg = cv2.bitwise_or(img2, img2, mask=mask_inv)
        pred_color = cv2.bitwise_or(bgr, fg)

        msg_out = self.bridge.cv2_to_imgmsg(confidence_img, "mono8")
        msg_out.header.stamp = img_msg.header.stamp

        pred_msg = self.bridge.cv2_to_imgmsg(pred_color, "bgr8")
        pred_msg.header.stamp = img_msg.header.stamp

        depth_msg = self.bridge.cv2_to_imgmsg(depth_bgr, "bgr8")
        depth_msg.header.stamp = img_msg.header.stamp

        self.pub_pred.publish(pred_msg)
        self.pub.publish(msg_out)
        self.pub_depth.publish(depth_msg)


def main(args):
    rospy.init_node("hanging_points_net", anonymous=False)
    hpn = HangingPointsNet()
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)
