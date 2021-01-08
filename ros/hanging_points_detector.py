#!/usr/bin/env python
# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import division

import copy
import os
import os.path as osp
import sys
import yaml

import cameramodels
import message_filters
import numpy as np
import rospkg
import rospy
import torch
from geometry_msgs.msg import PoseArray, Pose
from torchvision import transforms
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from skrobot import coordinates
from skrobot.coordinates.math import matrix2quaternion
from skrobot.coordinates.math import rotation_matrix_from_axis
from skrobot.coordinates.math import quaternion2matrix
from hanging_points_cnn.learning_scripts.hpnet import HPNET
from hanging_points_cnn.utils.image import colorize_depth
from hanging_points_cnn.utils.image import draw_axis
from hanging_points_cnn.utils.image import normalize_depth
from hanging_points_cnn.utils.image import unnormalize_depth
from hanging_points_cnn.utils.image import overlay_heatmap
from hanging_points_cnn.utils.image import remove_nan
from hanging_points_cnn.utils.rois_tools import make_box


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
        self.pub_confidence = rospy.Publisher(
            "~output/confidence", Image, queue_size=10)
        self.pub_depth = rospy.Publisher(
            "~colorized_depth", Image, queue_size=10)
        self.pub_axis = rospy.Publisher(
            "~axis", Image, queue_size=10)
        self.pub_axis_raw = rospy.Publisher(
            "~axis_raw", Image, queue_size=10)
        self.pub_hanging_points = rospy.Publisher(
            "/hanging_points", PoseArray, queue_size=10)

        self.gpu_id = rospy.get_param('~gpu', 0)
        self.predict_depth = rospy.get_param('~predict_depth', True)
        print('self.predict_depth: ', self.predict_depth)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        pretrained_model = rospy.get_param(
            '~pretrained_model',
            '/media/kosuke/SANDISK/hanging_points_net/checkpoints/gray/hpnet_bestmodel_20201025_1542.pt')  # noqa
        task_type = rospy.get_param('~task_type', 'hanging')
        config_path = rospy.get_param('~config', None)

        if config_path is None:
            rospack = rospkg.RosPack()
            pack_path = rospack.get_path('hanging_points_cnn')
            config_path = osp.join(
                pack_path,
                'hanging_points_cnn',
                'learning_scripts',
                'config',
                'gray_model.yaml')
        print('Load ' + config_path)
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.transform = transforms.Compose([
            transforms.ToTensor()])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.depth_range = self.config['depth_range']
        self.target_size = tuple(self.config['target_size'])
        self.depth_roi_size = self.config['depth_roi_size'][task_type]
        print('task type: {}'.format(task_type))
        print('depth roi size: {}'.format(self.depth_roi_size))

        self.model = HPNET(self.config).to(device)

        if osp.exists(pretrained_model):
            print('use pretrained model')
            self.model.load_state_dict(
                torch.load(pretrained_model), strict=False)

        self.model.eval()

        self.camera_model = None
        self.load_camera_info()

        self.use_coords = False

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
        self.camera_model.target_size = self.target_size

        bgr_raw = self.bridge.imgmsg_to_cv2(img_raw_msg, "bgr8")
        bgr = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        if cv_depth is None or bgr is None:
            return
        remove_nan(cv_depth)
        cv_depth[cv_depth < self.depth_range[0]] = 0
        cv_depth[cv_depth > self.depth_range[1]] = 0
        bgr = cv2.resize(bgr, self.target_size)
        cv_depth = cv2.resize(cv_depth, self.target_size,
                              interpolation=cv2.INTER_NEAREST)

        depth_bgr = colorize_depth(
            cv_depth, ignore_value=0)

        in_feature = cv_depth.copy().astype(np.float32) * 0.001

        if self.config['use_bgr2gray']:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, self.target_size)[..., None] / 255.
            normalized_depth = normalize_depth(
                cv_depth, self.depth_range[0], self.depth_range[1])[..., None]
            in_feature = np.concatenate(
                (normalized_depth, gray), axis=2).astype(np.float32)

        if self.transform:
            in_feature = self.transform(in_feature)

        in_feature = in_feature.to(self.device)
        in_feature = in_feature.unsqueeze(0)

        confidence, depth, rotation = self.model(in_feature)
        confidence = confidence[0, 0:1, ...]
        confidence_np = confidence.cpu().detach().numpy().copy() * 255
        confidence_np = confidence_np.transpose(1, 2, 0)
        confidence_np[confidence_np <= 0] = 0
        confidence_np[confidence_np >= 255] = 255
        confidence_img = confidence_np.astype(np.uint8)
        confidence_img = cv2.resize(confidence_img, self.target_size)
        heatmap = overlay_heatmap(bgr, confidence_img)

        axis_pred = bgr.copy()
        axis_pred_raw = bgr_raw.copy()

        dep_pred = []
        hanging_points_pose_array = PoseArray()
        for i, (roi, roi_center) in enumerate(
                zip(self.model.rois_list[0], self.model.rois_center_list[0])):
            if roi.tolist() == [0, 0, 0, 0]:
                continue
            roi = roi.cpu().detach().numpy().copy()
            hanging_point_x = roi_center[0]
            hanging_point_y = roi_center[1]
            v = rotation[i].cpu().detach().numpy()
            v /= np.linalg.norm(v)
            rot = rotation_matrix_from_axis(v, [0, 1, 0], 'xy')
            q = matrix2quaternion(rot)

            hanging_point = np.array(
                self.camera_model.project_pixel_to_3d_ray(
                    [int(hanging_point_x),
                     int(hanging_point_y)]))

            if self.predict_depth:
                dep = depth[i].cpu().detach().numpy().copy()
                dep = unnormalize_depth(
                    dep, self.depth_range[0], self.depth_range[1]) * 0.001
                length = float(dep) / hanging_point[2]
            else:
                depth_roi = make_box(
                    roi_center,
                    width=self.depth_roi_size[1],
                    height=self.depth_roi_size[0],
                    img_shape=self.target_size,
                    xywh=False)
                depth_roi_clip = cv_depth[
                    depth_roi[0]:depth_roi[2],
                    depth_roi[1]:depth_roi[3]]
                dep_roi_clip = depth_roi_clip[np.where(
                    np.logical_and(self.depth_range[0] < depth_roi_clip,
                                   depth_roi_clip < self.depth_range[1]))]
                dep_roi_clip = np.median(dep_roi_clip) * 0.001
                if dep_roi_clip == np.nan:
                    continue
                length = float(dep_roi_clip) / hanging_point[2]

            hanging_point *= length
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
                (int(roi[0] * (xmax - xmin) / self.target_size[1] + xmin),
                 int(roi[1] * (ymax - ymin) / self.target_size[0] + ymin)),
                (int(roi[2] * (xmax - xmin) / self.target_size[1] + xmin),
                 int(roi[3] * (ymax - ymin) / self.target_size[0] + ymin)),
                (0, 255, 0), 1)
            try:
                axis_pred_raw = draw_axis(axis_pred_raw,
                                          quaternion2matrix(q),
                                          hanging_point,
                                          self.camera_model.full_K)
            except Exception:
                print('Fail to draw axis')

        axis_pred = self.camera_model.crop_image(
            axis_pred_raw, copy=True).astype(np.uint8)

        msg_out = self.bridge.cv2_to_imgmsg(heatmap, "bgr8")
        msg_out.header.stamp = depth_msg.header.stamp

        confidence_msg = self.bridge.cv2_to_imgmsg(confidence_img, "mono8")
        confidence_msg.header.stamp = depth_msg.header.stamp

        colorized_depth_msg = self.bridge.cv2_to_imgmsg(depth_bgr, "bgr8")
        colorized_depth_msg.header.stamp = depth_msg.header.stamp

        axis_pred_msg = self.bridge.cv2_to_imgmsg(axis_pred, "bgr8")
        axis_pred_msg.header.stamp = depth_msg.header.stamp

        axis_pred_raw_msg = self.bridge.cv2_to_imgmsg(axis_pred_raw, "bgr8")
        axis_pred_raw_msg.header.stamp = depth_msg.header.stamp

        hanging_points_pose_array.header = camera_info_msg.header
        self.pub.publish(msg_out)
        self.pub_confidence.publish(confidence_msg)
        self.pub_depth.publish(colorized_depth_msg)
        self.pub_axis.publish(axis_pred_msg)
        self.pub_axis_raw.publish(axis_pred_raw_msg)
        self.pub_hanging_points.publish(hanging_points_pose_array)


def main(args):
    rospy.init_node("hanging_points_net", anonymous=False)
    HangingPointsNet()
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)
