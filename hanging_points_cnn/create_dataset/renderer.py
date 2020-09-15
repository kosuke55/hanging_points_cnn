#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import glob
import json
import numpy.matlib as npm
import os
import os.path as osp
import sys
from pathlib import Path
from PIL import Image

import cameramodels
import numpy as np
import pybullet
import pybullet_data
# import skrobot
import xml.etree.ElementTree as ET
from eos import make_fancy_output_dir
from sklearn.cluster import DBSCAN
from skrobot import coordinates

from hanging_points_generator.hp_generator import cluster_hanging_points
from hanging_points_generator.hp_generator import filter_penetration
from hanging_points_generator.generator_utils import get_urdf_center
from hanging_points_generator.generator_utils import load_multiple_contact_points
from hanging_points_cnn.utils.image import colorize_depth
from hanging_points_cnn.utils.image import create_circular_mask
from hanging_points_cnn.utils.image import create_depth_circle
from hanging_points_cnn.utils.image import create_gradient_circle
from hanging_points_cnn.utils.image import draw_axis


try:
    import cv2
except ImportError:
    for path in sys.path:
        if '/opt/ros/' in path:
            print('sys.path.remove({})'.format(path))
            sys.path.remove(path)
            import cv2
            sys.path.append(path)

np.set_printoptions(precision=3, suppress=True)


class Renderer:
    def __init__(
            self, im_width=512, im_height=424, fov=42.5,
            near_plane=0.1, far_plane=2.0, target_width=256, target_height=256,
            DEBUG=False):
        self.objects = []
        self.im_width = im_width
        self.im_height = im_height
        self.fov = fov
        self.near_plane = near_plane
        self.far_plane = far_plane
        aspect = self.im_width / self.im_height
        self.camera_model \
            = cameramodels.PinholeCameraModel.from_fov(
                fov, im_height, im_width)

        self.camera_model.target_size = (target_width, target_height)
        self.pm = pybullet.computeProjectionMatrixFOV(
            fov, aspect, near_plane, far_plane)

        self.camera_coords = coordinates.Coordinates(
            pos=np.array([0, 0, 0.5]),
            rot=coordinates.math.rotation_matrix_from_rpy([0, np.pi, 0]))

        self.annotation_img = np.zeros(
            (target_width, target_height), dtype=np.uint32)
        self.rotation_map = RotationMap(target_width, target_height)

        # self.object_pos = np.array([0, 0, 0.1])
        # self.object_rot = coordinates.math.rotation_matrix_from_rpy([0, 0, 0])

        self.object_coords = coordinates.Coordinates(
            pos=np.array([0, 0, 0.1]),
            rot=coordinates.math.rotation_matrix_from_rpy([0, 0, 0]))

        if DEBUG:
            self.cid = pybullet.connect(pybullet.GUI)
            pybullet.resetDebugVisualizerCamera(
                cameraDistance=1,
                cameraYaw=90,
                cameraPitch=0,
                cameraTargetPosition=[0, 0, 0.1])
        else:
            self.cid = pybullet.connect(pybullet.DIRECT)

        self.texture_paths = glob.glob(
            osp.join('/media/kosuke/SANDISK/dtd', '**', '*.jpg'),
            recursive=True)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setPhysicsEngineParameter(enableFileCaching=0)

        self.draw_camera_pos()
        self.change_light()
        self._rendered = None
        self._rendered_pos = None
        self._rendered_rot = None

    def load_urdf(self, urdf, random_pose=True):
        if random_pose:
            roll = np.random.rand() * np.pi
            pitch = np.random.rand() * np.pi
            yaw = np.random.rand() * np.pi
            self.object_id = pybullet.loadURDF(
                urdf, [0, 0, 0], pybullet.getQuaternionFromEuler(
                    [roll, pitch, yaw]))
        else:
            self.object_id = pybullet.loadURDF(
                urdf, [0, 0, 0], [0, 0, 0, 1])

        self.object_center = get_urdf_center(urdf)
        self.objects.append(self.object_id)
        pybullet.changeVisualShape(self.object_id, -1, rgbaColor=[1, 1, 1, 1])

        pos, rot = pybullet.getBasePositionAndOrientation(self.object_id)
        self.object_coords = coordinates.Coordinates(
            pos=pos, rot=coordinates.math.xyzw2wxyz(rot))

        return self.object_id

    def remove_object(self, o_id, update=True):
        pybullet.removeBody(o_id)
        if update:
            # print("remove ", o_id)
            self.objects.remove(o_id)

    def remove_all_objects(self):
        # objects = copy.copy(self.objects)
        # for o_id in objects:
        for o_id in self.objects:
            self.remove_object(o_id, True)
        self.objects = []

    def reset_object_pose(self, object_id):
        x = (np.random.rand() - 0.5) * 0.1
        y = (np.random.rand() - 0.5) * 0.1
        z = (np.random.rand() - 0.5) * 0.1
        roll = np.random.rand() * np.pi
        pitch = np.random.rand() * np.pi
        yaw = np.random.rand() * np.pi
        pybullet.setGravity(0, 0, 0)
        pybullet.resetBasePositionAndOrientation(
            object_id,
            [x, y, z],
            pybullet.getQuaternionFromEuler([roll, pitch, yaw]))
        return [x, y, z]

    def step(self, n=1):
        for i in range(n):
            pybullet.stepSimulation()

    def _rotation_matrix(self, rpy):
        r, p, y = rpy

        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(r), -np.sin(r), 0],
            [0, np.sin(r), np.cos(r), 0],
            [0, 0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(p), 0, np.sin(p), 0],
            [0, 1, 0, 0],
            [-np.sin(p), 0, np.cos(p), 0],
            [0, 0, 0, 1]
        ])
        Rz = np.array([
            [np.cos(y), -np.sin(y), 0, 0],
            [np.sin(y), np.cos(y), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return np.linalg.multi_dot([Rz, Ry, Rx])

    def draw_camera_pos(self):
        pybullet.removeAllUserDebugItems()
        start = self.camera_coords.worldpos()
        end_x = start + self.camera_coords.rotate_vector([0.1, 0, 0])
        pybullet.addUserDebugLine(start, end_x, [1, 0, 0], 3)
        end_y = start + self.camera_coords.rotate_vector([0, 0.1, 0])
        pybullet.addUserDebugLine(start, end_y, [0, 1, 0], 3)
        end_z = start + self.camera_coords.rotate_vector([0, 0, 0.1])
        pybullet.addUserDebugLine(start, end_z, [0, 0, 1], 3)

    def change_light(self):
        self.lightDirection = 10 * np.random.rand(3)
        self.lightDistance = 0.9 + 0.2 * np.random.rand()
        self.lightColor = 0.9 + 0.1 * np.random.rand(3)
        self.lightAmbientCoeff = 0.1 + 0.2 * np.random.rand()
        self.lightDiffuseCoeff = 0.85 + 0.1 * np.random.rand()
        self.lightSpecularCoeff = 0.85 + 0.1 * np.random.rand()

    def render(self):
        # If you comment this out, you cannot change the light state with the same pose.
        # if np.all(
        #         self._rendered_pos == self.camera_coords.worldpos()) and np.all(
        #         self._rendered_rot == self.camera_coords.worldrot()):
        #     return self._rendered

        target = self.camera_coords.worldpos() + \
            self.camera_coords.rotate_vector([0, 0, 1.])
        up = self.camera_coords.rotate_vector([0, -1.0, 0])
        # up = self.camera_coords.copy().rotate_vector([0, 1.0, 0])

        vm = pybullet.computeViewMatrix(
            self.camera_coords.worldpos(), target, up)

        i_arr = pybullet.getCameraImage(
            self.im_width, self.im_height, vm, self.pm,
            shadow=0,
            renderer=pybullet.ER_TINY_RENDERER,
            lightDirection=self.lightDirection,
            lightColor=self.lightColor,
            lightDistance=self.lightDistance,
            lightAmbientCoeff=self.lightAmbientCoeff,
            lightDiffuseCoeff=self.lightDiffuseCoeff,
            lightSpecularCoeff=self.lightSpecularCoeff
        )

        self._rendered = i_arr
        self._rendered_pos = self.camera_coords.worldpos().copy()
        self._rendered_rot = self.camera_coords.worldrot().copy()

        return i_arr

    def get_depth(self):
        return self.render()[3]

    def get_depth_metres(self, noise=0.001):
        d = self.render()[3]
        # Linearise to metres
        return 2 * self.far_plane * self.near_plane / (self.far_plane + self.near_plane - (
            self.far_plane - self.near_plane) * (2 * d - 1)) + np.random.randn(self.im_height, self.im_width) * noise

    def get_depth_milli_metres(self):
        self.depth = (self.get_depth_metres() * 1000).astype(np.float32)
        return self.depth

    def get_rgb(self):
        self.rgb = self.render()[2]
        return self.rgb

    def get_bgr(self):
        self.bgr = cv2.cvtColor(self.get_rgb(), cv2.COLOR_RGB2BGR)
        return self.bgr

    def get_seg(self):
        self.seg = self.render()[4]
        return self.seg

    def get_object_mask(self, object_id):
        if np.count_nonzero(self.seg == object_id) == 0:
            return None
        self.object_mask = np.where(self.seg == object_id)
        self.non_object_mask = np.where(self.seg != object_id)

        return self.object_mask, self.non_object_mask

    def get_object_depth(self):
        self.object_depth = r.get_depth_milli_metres()
        self.object_depth[self.non_object_mask] = 0
        return self.object_depth

    def get_roi(self, padding=0):
        ymin = np.max([np.min(self.object_mask[0]) -
                       np.random.randint(0, padding), 0])
        ymax = np.min([np.max(self.object_mask[0]) +
                       np.random.randint(0, padding),
                       int(self.im_height - 1)])
        xmin = np.max([np.min(self.object_mask[1]) -
                       np.random.randint(0, padding), 0])
        xmax = np.min([np.max(self.object_mask[1]) +
                       np.random.randint(0, padding),
                       int(self.im_width - 1)])
        # self.roi = [ymin, ymax, xmin, xmax]

        # [top, left, bottom, right] order
        self.camera_model.roi = [ymin, xmin, ymax, xmax]
        return [ymin, ymax, xmin, xmax]
        # return self.roi

    def get_visible_coords(self, contact_points, debug_line=False):
        self.hanging_point_in_camera_coords_list = []
        for cp in contact_points:
            hanging_point_coords = coordinates.Coordinates(
                pos=(cp[0] - self.object_center), rot=cp[1:])
            # hanging_point_coords.translate([0, 0.01, 0])
            hanging_point_worldcoords \
                = r.object_coords.copy().transform(
                    hanging_point_coords)
            hanging_point_in_camera_coords \
                = r.camera_coords.inverse_transformation(
                ).transform(hanging_point_worldcoords)
            # px, py = r.camera_model.project3d_to_pixel(
            #     hanging_point_in_camera_coords.worldpos())
            rayInfo = pybullet.rayTest(
                hanging_point_worldcoords.worldpos(),
                r.camera_coords.worldpos())
            if rayInfo[0][0] == r.camera_id:
                self.hanging_point_in_camera_coords_list.append(
                    hanging_point_in_camera_coords)
                pybullet.addUserDebugLine(
                    hanging_point_worldcoords.worldpos(),
                    r.camera_coords.worldpos(), [1, 1, 1], 1)
            # occulsion
            else:
                pybullet.addUserDebugLine(
                    hanging_point_worldcoords.worldpos(),
                    r.camera_coords.worldpos(), [1, 0, 0], 1)

        if len(self.hanging_point_in_camera_coords_list) == 0:
            print('-- No visible hanging point --')
            return False

        return self.hanging_point_in_camera_coords_list

    def move_to(self, T):
        self.camera_coords = coordinates.Coordinates(
            pos=np.array(T),
            rot=r.camera_coords.worldrot())
        pybullet.resetBasePositionAndOrientation(
            self.camera_id,
            self.camera_coords.worldpos(),
            coordinates.math.wxyz2xyzw(
                coordinates.math.matrix2quaternion(
                    self.camera_coords.worldrot())))

    def move_to_random_pos(self):
        newpos = [(np.random.rand() - 0.5) * 0.1,
                  (np.random.rand() - 0.5) * 0.1,
                  np.random.rand() * 0.7 + 0.3]
        self.move_to(newpos)

    def look_at(self, p):
        p = np.array(p)
        if np.all(p == self.camera_coords.worldpos()):
            return
        z = p - self.camera_coords.worldpos()
        coordinates.geo.orient_coords_to_axis(self.camera_coords, z)

        pybullet.resetBasePositionAndOrientation(
            self.camera_id,
            self.camera_coords.worldpos(),
            coordinates.math.wxyz2xyzw(
                coordinates.math.matrix2quaternion(
                    self.camera_coords.worldrot())))

        self.draw_camera_pos()

    def get_plane(self):
        self.plane_id = pybullet.loadURDF("plane.urdf", [0, 0, -0.5])
        return self.plane_id

    def save_intrinsics(self, save_dir):
        if not osp.isfile(
                osp.join(save_dir, 'intrinsics', 'intrinsics.npy')):
            np.save(osp.join(
                save_dir, 'intrinsics', 'intrinsics'), self.camera_model.K)

    def create_camera(self):
        camera_length = 0.01
        camera_object = pybullet.createCollisionShape(
            pybullet.GEOM_BOX,
            halfExtents=[camera_length, camera_length, camera_length])
        camera_object_visual = pybullet.createVisualShape(
            pybullet.GEOM_BOX,
            halfExtents=[camera_length,
                         camera_length, camera_length],
            rgbaColor=[0, 0, 0, 1])
        self.camera_id = pybullet.createMultiBody(
            baseMass=0.,
            baseCollisionShapeIndex=camera_object,
            baseVisualShapeIndex=camera_object_visual,
            basePosition=r.camera_coords.worldpos(),
            baseOrientation=[0, 0, 0, 1.])
        r.objects.append(self.camera_id)

        return self.camera_id

    def change_texture(self, object_id):
        textureId = pybullet.loadTexture(
            self.texture_paths[np.random.randint(
                0, len(self.texture_paths) - 1)])
        pybullet.changeVisualShape(
            object_id, -1, textureUniqueId=textureId)

    def finish(self):
        self.remove_all_objects()
        pybullet.resetSimulation()
        pybullet.disconnect()


class DepthMap():
    def __init__(self, width, height, circular=True):
        self.width = width
        self.height = height
        self.size = width * height
        self.idx_list = []
        self._depth_buffer = [[] for _ in range(self.size)]
        self._depth = [0] * depth.size
        self.circular = circular

    def add_depth(self, px, py, d):
        if self.circular:
            iy, ix = np.where(
                create_circular_mask(self.height, self.width, py, px))
            idices = ix + iy * self.width
            for idx in idices:
                self._depth_buffer[idx].append(d)
        else:
            idx = px + py * self.width
            self._depth[idx] = d

    def calc_average_depth(self):
        for idx in range(self.size):
            if self._depth_buffer[idx] != []:
                self._depth[idx] = np.mean(self._depth_buffer[idx])

    @property
    def depth(self):
        if self.circular:
            self.calc_average_depth()
        return np.array(self._depth).reshape(self.height, self.width)

    def on_depth_image(self, depth_image):
        depth_image = depth_image.copy()
        mask = np.where(self.depth != 0)
        depth_image[mask] = self.depth[mask]
        return depth_image


class RotationMap():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.size = width * height
        self.rotations_buffer = [[] for _ in range(self.size)]
        # self.rotations_buffer = []
        # [w, x, y, z] quaternion
        self.rotations = [np.array([1, 0, 0, 0])] * self.size

    def add_quaternion(self, px, py, q):
        iy, ix = np.where(
            create_circular_mask(self.height, self.width, py, px))
        idices = ix + iy * self.width
        for idx in idices:
            self.rotations_buffer[idx].append(q.tolist())

    def get_length(self, px, py):
        idx = px + py * self.width
        return len(self.rotations_buffer[idx])

    def averageQuaternions(self, Q):
        '''
        https://github.com/christophhagen/averaging-quaternions/blob/master/LICENSE
        Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
        The quaternions are arranged as (w,x,y,z), with w being the scalar
        The result will be the average quaternion of the input. Note that the signs
        of the output quaternion can be reversed, since q and -q describe the same orientation
        '''
        M = Q.shape[0]
        A = npm.zeros(shape=(4, 4))

        for i in range(0, M):
            q = Q[i, :]
            A = np.outer(q, q) + A

        A = (1.0 / M) * A
        eigenValues, eigenVectors = np.linalg.eig(A)
        eigenVectors = eigenVectors[:, eigenValues.argsort()[::-1]]

        # return the real part of the largest eigenvector (has only real part)
        return np.real(eigenVectors[:, 0].A1)

    def calc_average_rotations(self):
        for idx in range(self.size):
            if self.rotations_buffer[idx] != []:
                self.rotations[idx] = self.averageQuaternions(
                    np.array(self.rotations_buffer[idx]))


def get_contact_points(contact_points_path, json_name='contact_points.json',
                       dataset_type='ycb', use_clustering=True,
                       use_filter_penetration=True,
                       inf_penetration_check=True):

    if osp.isdir(contact_points_path):
        contact_points_dict = load_multiple_contact_points(
            contact_points_path, json_name)
    else:
        contact_points_dict = json.load(open(contact_points_path, 'r'))
    contact_points = contact_points_dict['contact_points']

    if use_clustering:
        contact_points = cluster_hanging_points(
            contact_points, min_samples=-1)

    if use_filter_penetration:
        if inf_penetration_check:
            box_size = [100, 0.0001, 0.0001]
        else:
            box_size = [0.1, 0.0001, 0.0001]

        if dataset_type == 'ycb':
            contact_points, _ = filter_penetration(
                osp.join(dirname, 'base.urdf'),
                contact_points, box_size=box_size)
        else:
            contact_points, _ = filter_penetration(
                osp.join(dirname, urdf_name),
                contact_points, box_size=box_size)

    if len(contact_points) == 0:
        print('num of hanging points: {}'.format(len(contact_points)))
        return None

    return contact_points


def make_save_dirs(save_dir):
    save_dir = make_fancy_output_dir(save_dir)
    os.makedirs(osp.join(save_dir, 'intrinsics'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'color'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'color_raw'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'debug_axis'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'debug_heatmap'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'depth'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'depth_bgr'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'hanging_points_depth'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'hanging_points_depth_bgr'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'heatmap'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'rotations'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'clip_info'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'camera_info'), exist_ok=True)
    return save_dir


def split_file_name(file, dataset_type='ycb'):
    dirname, filename = osp.split(file)
    filename_without_ext, ext = osp.splitext(filename)
    if dataset_type == 'ycb':
        category_name = dirname.split("/")[-1]
        idx = None
    elif dataset_type == 'ObjectNet3D':
        category_name = dirname.split("/")[-2]
        idx = dirname.split("/")[-1]
    return dirname, filename, category_name, idx


def align_coords(coords_list, eps=0.005, min_sample=2,
                 angle_thresh=135., copy_list=True):
    """Align the x-axis of coords

    invert coordinates above the threshold.
    If you do not align, the average value of the
    rotation map will be incorrect.

    Parameters
    ----------
    coords_list : list[skrobot.coordinates.base.Coordinates]
    eps : float, optional
        eps paramerter of sklearn dbscan, by default 0.005
    min_sample : int, optional
        min_sample paramerter of sklearn dbscan, by default 2
    angle_thresh : float, optional
        invert coordinates above the threshold, by default 135.0
    copy_list ; bool, optional
        If True copy coords_list, by default True

    Returns
    -------
    coords_list : list[skrobot.coordinates.base.Coordinates]
    """
    if copy_list:
        coords_list = copy.copy(coords_list)
    dbscan = DBSCAN(eps=eps, min_samples=min_sample).fit(
        [coords.worldpos() for coords in coords_list])

    for label in range(np.max(dbscan.labels_) + 1):
        if np.count_nonzero(dbscan.labels_ == label) <= 1:
            continue

        q_base = None

        for idx, coords in enumerate(coords_list):
            if dbscan.labels_[idx] == label:
                if q_base is None:
                    q_base = coords.quaternion
                q_distance \
                    = coordinates.math.quaternion_distance(
                        q_base, coords.quaternion)

                if np.rad2deg(q_distance) > angle_thresh:
                    coords_list[idx].rotate(np.pi, 'y')

    return coords_list


if __name__ == '__main__':
    print('Start')
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--save-dir', '-s',
        type=str, help='save dir',
        default='/media/kosuke/SANDISK-2/meshdata/ycb_hanging_object/0912_hoge')
    # default='ycb_hanging_object/per5000')
    parser.add_argument(
        '--data-num', '-n',
        type=int, help='num of data per object',
        default=1000)
    parser.add_argument(
        '--input-dir', '-i',
        type=str, help='input dir',
        default='/media/kosuke/SANDISK/meshdata/ycb_hanging_object/urdf2/')
    parser.add_argument(
        '--dataset-type', '-dt',
        type=str, help='dataset type',
        default='ycb')
    parser.add_argument(
        '--urdf-name', '-u',
        type=str, help='save dir',
        default='textured.urdf')
    parser.add_argument(
        '--gui', '-g',
        type=int, help='debug gui',
        default=0)
    parser.add_argument(
        '--show-image', '-si',
        type=int, help='show image',
        default=0)
    args = parser.parse_args()

    data_num = args.data_num
    dataset_type = args.dataset_type
    gui = args.gui
    input_dir = args.input_dir
    save_dir_base = args.save_dir
    urdf_name = args.urdf_name
    show_image = args.show_image

    file_paths = list(sorted(Path(
        input_dir).glob(osp.join('*', urdf_name))))
    files = list(map(str, file_paths))

    width = 256
    height = 256

    if dataset_type == 'ycb':
        category_name_list = [
            "019_pitcher_base",
            "022_windex_bottle",
            "025_mug",
            # "033_spatula", # no contact pointsa
            "035_power_drill",
            "037_scissors",
            "042_adjustable_wrench",
            "048_hammer",
            "050_medium_clamp",
            "051_large_clamp",
            "052_extra_large_clamp"
        ]
    elif dataset_type == 'ObjectNet3D':
        category_name_list = ['cup', 'key', 'scissors']

    try:
        for file in files:
            dirname, filename, category_name, idx \
                = split_file_name(file, dataset_type)
            save_dir = osp.join(save_dir_base, category_name)
            save_dir = make_save_dirs(save_dir)
            r = Renderer(DEBUG=False)
            r.save_intrinsics(save_dir)
            r.finish()

            # load multiple json
            contact_points = get_contact_points(
                osp.join(dirname, 'contact_points'))
            if contact_points is None:
                continue

            data_count = 0
            while data_count < data_num:
                r = Renderer(DEBUG=gui)
                r.get_plane()

                object_id = r.load_urdf(osp.join(dirname, urdf_name),
                                        random_pose=True)
                r.change_texture(object_id)
                r.change_texture(r.plane_id)
                r.create_camera()
                r.move_to_random_pos()
                r.look_at(r.object_coords.worldpos() - r.object_center)
                r.step(1)

                bgr = r.get_bgr()
                bgr_axis = bgr.copy()
                seg = r.get_seg()

                if r.get_object_mask(object_id) is None:
                    r.finish()
                    continue

                depth = r.get_object_depth()

                [ymin, ymax, xmin, xmax] = r.get_roi(padding=50)
                bgr = r.camera_model.crop_resize_image(bgr)
                bgr_axis = r.camera_model.crop_resize_image(bgr_axis)
                depth = r.camera_model.crop_resize_image(
                    depth, interpolation=Image.NEAREST)
                depth_bgr = colorize_depth(depth, 100, 1500)
                # depth_bgr = r.camera_model.crop_resize_image(depth_bgr)
                annotation_img = np.zeros(
                    r.camera_model.target_size, dtype=np.uint32)

                rotation_map = RotationMap(height, width)

                hanging_point_in_camera_coords_list \
                    = r.get_visible_coords(contact_points)
                if not hanging_point_in_camera_coords_list:
                    print('-- No visible hanging point --')
                    r.finish()
                    continue

                depth_map = DepthMap(width, height, circular=True)

                hanging_point_in_camera_coords_list \
                    = align_coords(hanging_point_in_camera_coords_list)

                for hp in hanging_point_in_camera_coords_list:
                    px, py = r.camera_model.project3d_to_pixel(hp.worldpos())
                    draw_axis(bgr_axis,
                              hp.worldrot(),
                              hp.worldpos(),
                              r.camera_model.K)
                    if 0 <= px <= width and 0 <= py <= height:
                        create_gradient_circle(
                            annotation_img,
                            int(py), int(px))

                        rotation_map.add_quaternion(
                            int(px), int(py), hp.quaternion)

                        depth_map.add_depth(
                            int(px), int(py),
                            hp.worldpos()[2] * 1000)

                if np.all(annotation_img == 0):
                    r.finish()
                    continue

                annotation_img = annotation_img / annotation_img.max() * 255

                annotation_img = annotation_img.astype(np.uint8)
                annotation_color = np.zeros_like(bgr, dtype=np.uint8)
                annotation_color[..., 2] = annotation_img
                bgr_annotation = cv2.addWeighted(bgr, 0.3,
                                                 annotation_color, 0.7, 0)

                rotation_map.calc_average_rotations()
                rotations = np.array(
                    rotation_map.rotations).reshape(height, width, 4)

                hanging_points_depth = depth_map.on_depth_image(depth)
                hanging_points_depth_bgr \
                    = colorize_depth(hanging_points_depth, ignore_value=0)

                if show_image:
                    cv2.imshow('annotation', annotation_img)
                    cv2.imshow('bgr', bgr)
                    cv2.imshow('bgr_annotation', bgr_annotation)
                    cv2.imshow('bgr_axis', bgr_axis)
                    cv2.imshow('depth_bgr', depth_bgr)
                    cv2.imshow('hanging_points_depth_bgr',
                               hanging_points_depth_bgr)
                    cv2.imshow('depth', depth)
                    cv2.imshow('hanging_points_depth',
                               hanging_points_depth)
                    cv2.waitKey(1)

                data_id = len(
                    glob.glob(osp.join(save_dir, 'depth', '*')))
                print('{}: {} sum: {}'.format(file, data_count, data_id))

                cv2.imwrite(
                    osp.join(
                        save_dir, 'color', '{:06}.png'.format(
                            data_id)), bgr)
                # cv2.imwrite(
                #     osp.join(
                #         save_dir, 'color_raw', '{:06}.png'.format(
                #             data_id)), bgr_raw)
                # cv2.imwrite(
                #     osp.join(
                #         save_dir, 'debug_axis', '{:06}.png'.format(
                #             data_id)), bgr_axis)
                # cv2.imwrite(
                #     osp.join(
                #         save_dir, 'debug_heatmap', '{:06}.png'.format(
                #             data_id)), bgr_annotation)
                np.save(
                    osp.join(
                        save_dir, 'depth', '{:06}'.format(data_id)), depth)
                np.save(
                    osp.join(
                        save_dir,
                        'hanging_points_depth', '{:06}'.format(data_id)),
                    hanging_points_depth)
                np.save(
                    osp.join(
                        save_dir, 'rotations', '{:06}'.format(data_id)),
                    rotations)
                # cv2.imwrite(
                #     osp.join(
                #         save_dir, 'depth_bgr', '{:06}.png'.format(
                #             data_id)), depth_bgr)
                # cv2.imwrite(
                #     osp.join(
                #         save_dir, 'hanging_points_depth_bgr', '{:06}.png'.format(
                #             data_id)), hanging_points_depth_bgr)
                cv2.imwrite(
                    osp.join(
                        save_dir, 'heatmap', '{:06}.png'.format(
                            data_id)), annotation_img)
                r.camera_model.dump(osp.join(
                    save_dir, 'camera_info', '{:06}.yaml'.format(
                        data_id)))
                data_count += 1
                # data_id += 1
                r.finish()

            # r.remove_all_objects()
            # pybullet.resetSimulation()
            # pybullet.disconnect()

    except KeyboardInterrupt:
        sys.exit()
