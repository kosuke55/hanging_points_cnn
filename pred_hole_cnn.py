#!/usr/bin/env python
# -*- coding: utf-8 -*

""".
Project 3d pointcloud to pixel.
"""

import image_geometry
import rospy
from sensor_msgs.msg import CameraInfo
import sys
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from skimage.morphology import convex_hull_image

import numpy as np
import cv2
import torch
import torch.optim as optim
import visdom
from NuscData import test_dataloader, train_dataloader
from weighted_mse import wmse
from BCNN import BCNN
from torchvision import transforms



class Holecnn():
    def __init__(self):
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("~output", Image, queue_size=10)
        self.pub_pred = rospy.Publisher("~output/pred", Image, queue_size=10)
        self.input_image = rospy.get_param(
            '~image',
            '/apply_mask_image_in_gripper/output')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_model = rospy.get_param(
            '~pretrained_model',
            'checkpoints/latest_model_clip.pt')
        # pretrained_model = rospy.get_param(
        #     '~pretrained_model',
        #     'checkpoints/mango_best.pt')
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        # self.model = BCNN()
        self.model = BCNN().to(self.device)
        self.model.load_state_dict(torch.load(pretrained_model))
        self.model.eval()
        self.subscribe()

    def subscribe(self):
        self.image_sub = rospy.Subscriber(
            self.input_image, Image, self.callback, queue_size=1)

    def callback(self, img_msg):
        img = self.bridge.imgmsg_to_cv2(img_msg, "rgb8")
        img = cv2.resize(img, (640, 640))
        pred_img = img.copy()
        # print("0  ", img.shape)
        # img = torch.FloatTensor(img.astype(np.float32)).to(self.device)
        if self.transform:
            img = self.transform(img)
        img = torch.tensor(img, requires_grad=True)
        img = img.to(self.device)
        # img = transforms.ToTensor(img.astype(np.float32))
        # print("1  ", img.shape)
        img = img.unsqueeze(0)
        print("2  ", img.shape)
        output = self.model(img)
        output = output[:, 0, :, :]
        # output = torch.sigmoid(output)
        output_np = output.cpu().detach().numpy().copy()
        output_np = output_np.transpose(1, 2, 0)
        # print(output_np)
        output_img = output_np * 255
        output_img = output_img.astype(np.uint8)

        # pred_img = pred_img[0, ...]
        # pred_img = pred_img.transpose(1, 2, 0)
        conf_idx = np.where(output_np[..., 0] > 0.01)
        pred_img[conf_idx] = [255, 0, 0]

        msg_out = self.bridge.cv2_to_imgmsg(output_img, "mono8")
        msg_out.header.stamp = img_msg.header.stamp

        pred_msg = self.bridge.cv2_to_imgmsg(pred_img, "rgb8")
        pred_msg.header.stamp = img_msg.header.stamp
        print(output.shape)
        self.pub.publish(msg_out)
        self.pub_pred.publish(pred_msg)


def main(args):
    rospy.init_node("holecnn", anonymous=False)
    holecnn = Holecnn()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)

