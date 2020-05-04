#!/usr/bin/env python
# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import division

import os
import sys

import image_geometry
import rospy
from sensor_msgs.msg import CameraInfo
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from skimage.morphology import convex_hull_image

import message_filters
import numpy as np
import cv2
import torch
import torch.optim as optim
import os
import os.path as osp
import visdom
# from BCNN import BCNN
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from learning_scripts.UNET import UNET
from torchvision import transforms


class HangingPointsNet():
    def __init__(self):
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("~output", Image, queue_size=10)
        self.pub_pred = rospy.Publisher("~output/pred", Image, queue_size=10)
        self.pub_depth = rospy.Publisher("~colorized_depth", Image, queue_size=10)
        self.input_image = rospy.get_param(
            '~image',
            '/apply_mask_image/output')
        self.input_depth = rospy.get_param(
            '~image',
            '/apply_mask_depth/output')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_model = rospy.get_param(
            '~pretrained_model',
            '../learning_scripts/checkpoints/unet_bestmodel_20200504_2253.pt')
        # pretrained_model = rospy.get_param(
        #     '~pretrained_model',
        #     'checkpoints/mango_best.pt')
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        # self.model = BCNN()
        # self.model = BCNN().to(self.device)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNET(in_channels=3).to(device)
        if os.path.exists(pretrained_model):
            print('use pretrained model')
            self.model.load_state_dict(torch.load(pretrained_model))
        self.model.eval()
        self.subscribe()

    def subscribe(self):
        self.sub_rgb = message_filters.Subscriber(
            self.input_image, Image, queue_size=1, buff_size=2**24
        )
        self.sub_depth = message_filters.Subscriber(
            self.input_depth, Image, queue_size=1, buff_size=2**24
        )
        sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth],
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

    def callback(self, img_msg, depth_msg):
        # bgr = self.bridge.imgmsg_to_cv2(img_msg, "rgb8")
        bgr = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

        if depth is None:
            return

        print(np.max(depth), np.min(depth))
        depth = self.colorize_depth(depth)
        depth_rgb = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)

        bgr = cv2.resize(bgr, (256, 256))
        depth = cv2.resize(depth, (256, 256))
        # import pdb
        # pdb.set_trace()
        depth[np.where(np.all(bgr==[0, 0, 0], axis=-1))] = [0, 0, 0]
        # img = np.concatenate((bgr, depth), axis=2).astype(np.float32)
        img = depth
        # pred_img = img.copy()
        # print("0  ", img.shape)
        # img = torch.FloatTensor(img.astype(np.float32)).to(self.device)
        print(img.shape)
        if self.transform:
            img = self.transform(img)


        print(img.shape)
        # img = torch.tensor(img, requires_grad=True)
        img = img.to(self.device)
        # img = transforms.ToTensor(img.astype(np.float32))
        # print("1  ", img.shape)
        img = img.unsqueeze(0)
        print("2  ", img.shape)
        # output = self.model(img)
        output = self.model(img)
        output = output[:, 0, :, :]
        # output = torch.sigmoid(output)
        output_np = output.cpu().detach().numpy().copy()
        output_np = output_np.transpose(1, 2, 0)
        # print(output_np)
        # output_img = output_np * 255
        output_np[output_np <= 0] = 0
        # output_np /= np.max(output_np)
        # output_np *= 255
        output_np[output_np >= 255] = 255
        output_img = output_np.astype(np.uint8)

        # annotation_img = annotation_img.astype(np.uint8)
        print(output_img.shape)
        pred_color = np.zeros_like(bgr, dtype=np.uint8)
        pred_color[..., 2] = output_img[..., 0]

        img2 = pred_color
        img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        mask_inv = cv2.bitwise_not(img2gray)
        white_background = np.full(img2.shape, 255, dtype=np.uint8)
        bk = cv2.bitwise_or(white_background, white_background, mask=mask_inv)
        fg = cv2.bitwise_or(img2, img2, mask=mask_inv)
        pred_color = cv2.bitwise_or(bgr,fg)
        
        # pred_color = cv2.addWeighted(bgr, 0.3,
        #                              pred_color, 0.7, 0)
        # pred_color = cv2.LUT(pred_color,gamma_cvt)
        # pred_img = pred_img[0, ...]
        # pred_img = pred_img.transpose(1, 2, 0)
        # conf_idx = np.where(output_np[..., 0] > 0.01)
        # pred_img[conf_idx] = [255, 0, 0]

        msg_out = self.bridge.cv2_to_imgmsg(output_img, "mono8")
        msg_out.header.stamp = img_msg.header.stamp

        pred_msg = self.bridge.cv2_to_imgmsg(pred_color, "bgr8")
        # pred_msg = self.bridge.cv2_to_imgmsg(bgr, "rgb8")        
        pred_msg.header.stamp = img_msg.header.stamp

        # depth_msg = self.bridge.cv2_to_imgmsg(depth, "rgb8")
        depth_msg = self.bridge.cv2_to_imgmsg(depth, "bgr8")
        depth_msg.header.stamp = img_msg.header.stamp
        print(output.shape)
        self.pub_pred.publish(pred_msg)
        self.pub.publish(msg_out)
        self.pub_depth.publish(depth_msg)


def main(args):
    rospy.init_node("hanging_points_net", anonymous=False)
    hpn = HangingPointsNet()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)

