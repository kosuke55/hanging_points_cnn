#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import argparse
import copy
import glob
import json
import os
import os.path as osp
import random
import shutil
import sys
from operator import itemgetter
from pathlib import Path
from PIL import Image

import cameramodels
import numpy as np
import numpy.matlib as npm
import pybullet
import pybullet_data
from eos import make_fancy_output_dir
from sklearn.cluster import DBSCAN
from skrobot import coordinates

from hanging_points_generator.create_mesh import load_camera_pose
from hanging_points_generator.generator_utils import cluster_contact_points
from hanging_points_generator.generator_utils import filter_penetration
from hanging_points_generator.generator_utils import get_urdf_center
from hanging_points_generator.generator_utils import load_list
from hanging_points_generator.generator_utils import save_json
from hanging_points_cnn.utils.image import create_circular_mask
from hanging_points_cnn.utils.image import create_gradient_circle
from hanging_points_cnn.utils.image import draw_axis

from skrobot.coordinates.math import rotation_matrix_from_axis

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
            self, im_width=512,
            im_height=424, fov=42.5,
            near_plane=0.1, far_plane=30.0,
            target_width=256, target_height=256,
            use_change_light=True, labels=None,
            save_dir='./', save_debug_image=False,
            gui=False, task_type='hanging', stop_per_data=False):
        """Create training data of CNN

        Parameters
        ----------
        im_width : int, optional
            sim camera width, by default 512
        im_height : int, optional
            sim camera height, by default 424
        fov : float, optional
            sim camera fov, by default 42.5
        near_plane : float, optional
            by default 0.1
        far_plane : float, optional
            by default 2.0
        target_width : int, optional
            created data width, by default 256
        target_height : int, optional
             created data height, by default 256
        save_dir : str, optional
            [description], by default './'
        gui : bool, optional
            enable gui , by default False
        task_type : str, optional
            hanging or pouring.
            the value of translate depends on task_type.
            if task_type is hanging, tranlate value is [0, 0.01, 0]
        """
        self.objects = []
        self.im_width = im_width
        self.im_height = im_height
        self.fov = fov
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.target_width = target_width
        self.target_height = target_height
        self.save_dir = save_dir
        self.save_debug_image = save_debug_image
        self.task_type = task_type
        self.stop_per_data = stop_per_data

        if self.task_type == 'hanging':
            # direction of grabity
            self.translate_value = np.array([0, 0.005, 0])
        elif self.task_type == 'pouring':
            # direction opposite to gravity
            self.translate_value = np.array([-0.005, 0, 0])

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

        self.annotation_data = []
        self.rotation_map = RotationMap(target_width, target_height)
        self.rotations = None
        self.depth_map = DepthMap(target_width, target_height, circular=True)

        self.object_coords = coordinates.Coordinates(
            pos=np.array([0, 0, 0.1]),
            rot=coordinates.math.rotation_matrix_from_rpy([0, 0, 0]))

        self.labels = labels
        self.visible_labels = []

        if gui:
            self.cid = pybullet.connect(pybullet.GUI)
            pybullet.resetDebugVisualizerCamera(
                cameraDistance=1,
                cameraYaw=90,
                cameraPitch=0,
                cameraTargetPosition=[0, 0, 0.1])
        else:
            self.cid = pybullet.connect(pybullet.DIRECT)
        self.gui = gui
        self.no_visible_count = 0
        self.no_visible_skip_num = 30

        self.texture_paths = list(
            map(str, list(Path('/media/kosuke/SANDISK/dtd').glob('**/*.jpg'))))

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setPhysicsEngineParameter(enableFileCaching=0)

        self.draw_camera_pos()
        self.lightDirection = [1, 1, 1]
        self.lightDistance = 1
        self.lightColor = [1, 1, 1]
        self.lightAmbientCoeff = 0.2
        self.lightDiffuseCoeff = 0.9
        self.lightSpecularCoeff = 0.9
        if use_change_light:
            self.change_light()
        self._rendered = None
        self._rendered_pos = None
        self._rendered_rot = None

    def load_urdf(self, urdf, pos=[0, 0, 0], rot=[1, 0, 0, 0],
                  random_pose=True, random_texture=True):
        """Load urdf

        Parameters
        ----------
        urdf : str
        random_pose : bool, optional
            If true, rotate object to random pose, by default True
        pos ; list[float]
        rot : list[float]
        Returns
        -------
        self.object_id : int
        """
        c = coordinates.Coordinates(pos=pos, rot=rot)
        rot = coordinates.math.wxyz2xyzw(c.quaternion)
        self.urdf_file = urdf
        self.object_id = pybullet.loadURDF(urdf, pos, rot)
        if random_pose:
            self.reset_object_pose()

        self.object_center = get_urdf_center(urdf)
        self.objects.append(self.object_id)
        if random_texture:
            pybullet.changeVisualShape(
                self.object_id, -1, rgbaColor=[1, 1, 1, 1])

        pos, rot = pybullet.getBasePositionAndOrientation(self.object_id)
        self.object_coords = coordinates.Coordinates(
            pos=pos, rot=coordinates.math.xyzw2wxyz(rot))

        return self.object_id

    def remove_object(self, o_id, update=True):
        """Remove object

        Parameters
        ----------
        o_id : int
            remove target object id
        update : bool, optional
            remove form self.objects list, by default True
        """
        pybullet.removeBody(o_id)
        if update:
            # print("remove ", o_id)
            self.objects.remove(o_id)

    def remove_all_objects(self):
        """Remove all objects"""
        # objects = copy.copy(self.objects)
        # for o_id in objects:
        for o_id in self.objects:
            self.remove_object(o_id, True)
        self.objects = []

    def move_object_coords(self, coords):
        """Move object to target coords

        Parameters
        ----------
        coords : skrobot.coordinates.base.Coordinates
        """
        self.object_coords = coords
        pybullet.resetBasePositionAndOrientation(
            self.object_id,
            self.object_coords.worldpos(),
            coordinates.math.wxyz2xyzw(
                coordinates.math.matrix2quaternion(
                    self.object_coords.worldrot())))

    def reset_object_pose(self):
        """Reset object rotation randomly"""
        roll = np.random.rand() * np.pi * 2
        pitch = np.random.rand() * np.pi * 2
        yaw = np.random.rand() * np.pi * 2
        pybullet.resetBasePositionAndOrientation(
            self.object_id,
            [0, 0, 0],
            pybullet.getQuaternionFromEuler([roll, pitch, yaw]))
        pos, rot = pybullet.getBasePositionAndOrientation(self.object_id)
        self.object_coords = coordinates.Coordinates(
            pos=pos, rot=coordinates.math.xyzw2wxyz(rot))

    def step(self, n=1):
        """Step simulation

        Parameters
        ----------
        n : int, optional
            the number of step, by default 1
        """
        for i in range(n):
            pybullet.stepSimulation()

    def draw_camera_pos(self):
        """Draw camera pose axis"""
        pybullet.removeAllUserDebugItems()
        start = self.camera_coords.worldpos()
        end_x = start + self.camera_coords.rotate_vector([0.1, 0, 0])
        pybullet.addUserDebugLine(start, end_x, [1, 0, 0], 3)
        end_y = start + self.camera_coords.rotate_vector([0, 0.1, 0])
        pybullet.addUserDebugLine(start, end_y, [0, 1, 0], 3)
        end_z = start + self.camera_coords.rotate_vector([0, 0, 0.1])
        pybullet.addUserDebugLine(start, end_z, [0, 0, 1], 3)

    def change_light(self):
        """Change light condition"""
        self.lightDirection = 10 * np.random.rand(3)
        self.lightDistance = 0.9 + 0.2 * np.random.rand()
        self.lightColor = 0.9 + 0.1 * np.random.rand(3)
        self.lightAmbientCoeff = 0.1 + 0.2 * np.random.rand()
        self.lightDiffuseCoeff = 0.85 + 0.1 * np.random.rand()
        self.lightSpecularCoeff = 0.85 + 0.1 * np.random.rand()

    def render(self):
        """Render and get color, depth and segmentation image"""

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
        """Get depth

        Returns
        -------
        depth_buffer: numpy.ndarray
        """
        return self.render()[3]

    def get_depth_metres(self, noise=0.001):
        """Get depth metres

        Parameters
        ----------
        noise : float, optional
            by default 0.001

        Returns
        -------
        depth_meres : numpy.ndarray
        """
        d = self.render()[3]
        # Linearise to metres
        return 2 * self.far_plane * self.near_plane / (self.far_plane + self.near_plane - (
            self.far_plane - self.near_plane) * (2 * d - 1)) + np.random.randn(self.im_height, self.im_width) * noise

    def get_depth_milli_metres(self):
        """Get depth metres

        Returns
        -------
        depth_milli_meres : numpy.ndarray
        """
        self.depth = (self.get_depth_metres() * 1000).astype(np.float32)
        self.full_depth = copy.copy(self.depth)
        return self.depth

    def get_rgb(self):
        """Get rgb

        Returns
        -------
        rgb : numpy.ndarray
        """
        self.rgb = self.render()[2]
        self.full_rgb = self.rgb.copy()
        return self.rgb

    def get_bgr(self):
        """Get bgr

        Returns
        -------
        bgr : numpy.ndarray
        """
        self.bgr = cv2.cvtColor(self.get_rgb(), cv2.COLOR_RGB2BGR)
        self.full_bgr = self.bgr.copy()
        return self.bgr

    def get_seg(self):
        """Get segmentation image

        Returns
        -------
        seg : numpy.ndarray
        """
        self.seg = self.render()[4]
        return self.seg

    def get_object_mask(self, object_id):
        """Get mask image of object

        Parameters
        ----------
        object_id : int
            target object id

        Returns
        -------
        mask : numpy.ndarray
        """
        if np.count_nonzero(self.seg == object_id) == 0:
            return None
        self.object_mask = np.where(self.seg == object_id)
        self.non_object_mask = np.where(self.seg != object_id)

        return self.object_mask, self.non_object_mask

    def get_object_depth(self):
        """Get depth of object

        Returns
        -------
        depth : numpy.ndarray
        """
        self.object_depth = self.get_depth_milli_metres()
        self.object_depth[self.non_object_mask] = 0
        return self.object_depth

    def get_roi(self, padding=0, random=True):
        """Get roi from object mask

        Parameters
        ----------
        padding : int, optional
            by default 0

        Returns
        -------
        roi : list[float]
            [top, left, bottom, right] order
        """
        if random:
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
        else:
            ymin = np.max([np.min(self.object_mask[0]) - padding, 0])
            ymax = np.min([np.max(self.object_mask[0]) +
                           padding, int(self.im_height - 1)])
            xmin = np.max([np.min(self.object_mask[1]) - padding, 0])
            xmax = np.min([np.max(self.object_mask[1]) +
                           padding, int(self.im_width - 1)])

        self.camera_model.roi = [ymin, xmin, ymax, xmax]
        self.roi_image = self.camera_model.draw_roi(
            self.bgr, copy=True)  # bgr is full before crop
        return [ymin, ymax, xmin, xmax]

    def transform_contact_points(self, contact_points_coords,
                                 translate=[0, 0.01, 0]):
        """Transform contact points

        Parameters
        ----------
        contact_points : list[list[list[float], list[float]]]

        contact_points_coords : list[skrobot.coordinates.Coordinates]
            [pos, rot] order
        translate : list, optional
            translate contact points for ray-trace,
            by default [0, 0.01, 0]

        Returns
        -------
        contact_points : list[list[list[], list[]]]
        """
        contact_point_worldcoords_list = []
        contact_point_in_camera_coords_list = []
        for contact_point_coords in contact_points_coords:
            contact_point_worldcoords \
                = self.object_coords.copy().transform(
                    contact_point_coords.translate(translate))
            contact_point_worldcoords_list.append(contact_point_worldcoords)
            contact_point_in_camera_coords \
                = self.camera_coords.inverse_transformation(
                ).transform(contact_point_worldcoords)
            contact_point_in_camera_coords_list.append(
                contact_point_in_camera_coords)

        return contact_point_in_camera_coords_list, \
            contact_point_worldcoords_list

    def make_contact_points_coords(self, contact_points):
        """Make contact points coords

        Parameters
        ----------
        contact_points : list[list[list[float], list[float]]]

        Returns
        -------
        contact_points_coords:  list[skrobot.coordinates.Coordinates]
        """
        contact_points_coords = []
        for cp in contact_points:
            contact_point_coords = coordinates.Coordinates(
                pos=(cp[0] - self.object_center), rot=cp[1:])
            contact_points_coords.append(contact_point_coords)
        return contact_points_coords

    def coords_to_dict(self, coords_list, translate=True):
        """Cover coords list to dict for json

        Parameters
        ----------
        coords_list : list[skrobot.coordinates.Coordinates]

        Returns
        -------
        contact_points_dict : dict
        """
        contact_points_list = []
        contact_points_dict = {
            'urdf_file': self.urdf_file,
            'contact_points': []}
        for coords in coords_list:
            if translate:
                coords = coords.copy_worldcoords().translate(
                    self.object_center, 'world')
            pose = np.concatenate(
                [coords.T()[:3, 3][None, :],
                 coords.T()[:3, :3]]).tolist()
            contact_points_list.append(pose)
        contact_points_dict['contact_points'] = contact_points_list
        return contact_points_dict

    def dbscan_coords(self, coords_list, eps=0.01, min_sample=2):
        self.dbscan = DBSCAN(eps=eps, min_samples=min_sample).fit(
            [coords.worldpos() for coords in coords_list])

    def align_coords(self, coords_list, eps=0.03,
                     min_sample=2, angle_thresh=90., copy_list=True):
        """Align the x-axis of coords

        invert coordinates above the threshold.
        If you do not align, the average value of the
        rotation map will be incorrect.

        Parameters
        ----------
        coords_list : list[skrobot.coordinates.base.Coordinates]
        eps : float, optional
            eps paramerter of cdrn dbscan, by default 0.005
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

        # dbscan = DBSCAN(eps=eps, min_samples=min_sample).fit(
        #     [coords.worldpos() for coords in coords_list])
        self.dbscan_coords(coords_list, eps, min_sample)
        for label in range(np.max(self.dbscan.labels_) + 1):
            q_base = None
            for idx, coords in enumerate(coords_list):
                if self.dbscan.labels_[idx] == label:
                    if q_base is None:
                        q_base = coords.quaternion
                    q_distance \
                        = coordinates.math.quaternion_distance(
                            q_base, coords.quaternion)

                    if np.rad2deg(q_distance) > angle_thresh:
                        coords_list[idx].rotate(np.pi, 'y')

        return coords_list

    def split_label_coords(self, coords_list):
        """Split coords based on label

        Parameters
        ----------
        coords_list : list[skrobot.coordinates.Coordinates]

        Returns
        -------
        coords_clusters :list[list[skrobot.coordinates.Coordinates]]
        """
        coords_clusters = []

        for label in range(np.max(self.dbscan.labels_) + 1):
            idx = tuple(np.where(self.dbscan.labels_ == label)[0])
            coords_cluster = itemgetter(*idx)(coords_list)

            coords_clusters.append(coords_cluster)
        return coords_clusters

    def make_average_coords_list(self, coords_list):
        """Make average coords list

        Parameters
        ----------
        coords_list : list[skrobot.coordinates.Coordinates]

        Returns
        -------
        coords_list : list[skrobot.coordinates.Coordinates]
        """
        average_coords_list = []
        coords_clusters = self.split_label_coords(coords_list)
        for coords_cluster in coords_clusters:
            coords_average = average_coords(coords_cluster)
            for coords in coords_cluster:
                coords.rotation = coords_average.rotation
                average_coords_list.append(coords)

        return average_coords_list

    def get_visible_coords(self, contact_points_coords,
                           debug_line=False, all_visible=False,
                           translate=[0, 0.01, 0]):
        """Transform and get visible coords

        Parameters
        ----------
        contact_points_coords : list[skrobot.coordinates.Coordinates]
        debug_line : bool, optional
            visualize debug line from cam to points, by default False
        all_visible : bool, optional
            If True, just transform coords and don't filter non-visible points.
            by default False
        translate : list, optional
            translate contact points for ray-trace,
            by default [0, 0.01, 0]. this is for hanging.

        Returns
        -------
        self.hanging_point_in_camera_coords_list
            : list[skrobot.coordinates.Coordinates]
            visible coords list
        """
        self.hanging_point_in_camera_coords_list = []
        contact_point_in_camera_coords_list, contact_point_worldcoords_list \
            = self.transform_contact_points(
                contact_points_coords, translate=translate)
        ray_info_list = pybullet.rayTestBatch(
            [c.worldpos() for c in contact_point_worldcoords_list],
            [self.camera_coords.worldpos()] * len(
                contact_point_in_camera_coords_list))
        for i, (coords_w, coords_c, ray_info) in enumerate(zip(
                contact_point_worldcoords_list,
                contact_point_in_camera_coords_list,
                ray_info_list)):
            if ray_info[0] == self.camera_id or all_visible:
                self.hanging_point_in_camera_coords_list.append(coords_c)
                if self.labels is not None:
                    self.visible_labels.append(self.labels[i])
                if self.gui:
                    pybullet.addUserDebugLine(
                        coords_w.worldpos(),
                        self.camera_coords.worldpos(), [1, 1, 1], 3)
            else:
                if self.gui:
                    pybullet.addUserDebugLine(
                        coords_w.worldpos(),
                        self.camera_coords.worldpos(), [1, 0, 0], 3)

        if len(self.hanging_point_in_camera_coords_list) == 0:
            print('-- No visible hanging point --')
            return False
        else:
            print('-- Find visible hanging point --')
        return self.hanging_point_in_camera_coords_list

    def move_to_coords(self, coords):
        """Move camera to target coords

        Parameters
        ----------
        coords : skrobot.coordinates.base.Coordinates
        """
        self.camera_coords = coords
        pybullet.resetBasePositionAndOrientation(
            self.camera_id,
            self.camera_coords.worldpos(),
            coordinates.math.wxyz2xyzw(
                coordinates.math.matrix2quaternion(
                    self.camera_coords.worldrot())))
        self.draw_camera_pos()

    def move_to(self, T):
        """Move camera to target position

        Parameters
        ----------
        T : list[float]
            move to [x, y ,z]
        """
        self.camera_coords = coordinates.Coordinates(
            pos=np.array(T),
            rot=self.camera_coords.worldrot())
        pybullet.resetBasePositionAndOrientation(
            self.camera_id,
            self.camera_coords.worldpos(),
            coordinates.math.wxyz2xyzw(
                coordinates.math.matrix2quaternion(
                    self.camera_coords.worldrot())))

    def move_to_random_pos(self):
        """Move camera to random position"""
        newpos = [(np.random.rand() - 0.5) * 0.1,
                  (np.random.rand() - 0.5) * 0.1,
                  np.random.rand() * 0.9 + 0.2]
        self.move_to(newpos)

    def look_at(self, p, horizontal=False):
        """Look at target position

        Parameters
        ----------
        p : list[float]
            look at [x, y, z]
        """
        p = np.array(p)
        if np.all(p == self.camera_coords.worldpos()):
            return
        z = p - self.camera_coords.worldpos()
        coordinates.geo.orient_coords_to_axis(self.camera_coords, z)
        if horizontal:
            self.camera_coords.newcoords(
                coordinates.Coordinates(
                    pos=self.camera_coords.worldpos(),
                    rot=rotation_matrix_from_axis(z, [0, 0, -1], axes='zy')))

        pybullet.resetBasePositionAndOrientation(
            self.camera_id,
            self.camera_coords.worldpos(),
            coordinates.math.wxyz2xyzw(
                coordinates.math.matrix2quaternion(
                    self.camera_coords.worldrot())))

        self.draw_camera_pos()

    def get_plane(self):
        """Load plane

        Returns
        -------
        self.plane_id : int
        """
        self.plane_id = pybullet.loadURDF("plane.urdf", [0, 0, -0.5])
        return self.plane_id

    def save_intrinsics(self, save_dir):
        """Save intrinsics

        Saving cameara info is better.

        Parameters
        ----------
        save_dir : str
        """
        if not osp.isfile(
                osp.join(save_dir, 'intrinsics', 'intrinsics.npy')):
            np.save(osp.join(
                save_dir, 'intrinsics', 'intrinsics'), self.camera_model.K)

    def create_camera(self):
        """Create camera object

        Returns
        -------
        self.camera_id : int
        """
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
            basePosition=self.camera_coords.worldpos(),
            baseOrientation=[0, 0, 0, 1.])
        self.objects.append(self.camera_id)

        return self.camera_id

    def change_texture(self, object_id):
        """Chage textuere of object

        Parameters
        ----------
        object_id : int
        """
        textureId = pybullet.loadTexture(
            self.texture_paths[np.random.randint(
                0, len(self.texture_paths) - 1)])
        pybullet.changeVisualShape(
            object_id, -1, textureUniqueId=textureId)

    def crop(self, padding, random=True):
        """Crop bgr and depth using object mask"""
        self.get_roi(padding=padding, random=random)
        self.bgr = self.camera_model.crop_resize_image(self.bgr)
        self.depth = self.camera_model.crop_resize_image(
            self.depth, interpolation=Image.NEAREST)

    def create_annotation_data(self):
        """Get annotation data

        Returns
        -------
        result : bool
        """
        for i, hp in enumerate(self.hanging_point_in_camera_coords_list):
            px, py = self.camera_model.project3d_to_pixel(hp.worldpos())
            if self.save_debug_image:
                self.bgr_axis = self.bgr.copy()
            if 0 <= px < self.target_width and 0 <= py < self.target_height:
                if self.save_debug_image:
                    draw_axis(self.bgr_axis,
                              hp.worldrot(),
                              hp.worldpos(),
                              self.camera_model.K)
                create_gradient_circle(
                    self.annotation_img,
                    int(py), int(px))
                if self.visible_labels == []:
                    self.annotation_data.append(
                        {'xy': [int(px), int(py)],
                         'depth': hp.worldpos()[2] * 1000,
                         'quaternion': hp.quaternion.tolist()}
                    )
                else:
                    self.annotation_data.append(
                        {'xy': [int(px), int(py)],
                         'depth': hp.worldpos()[2] * 1000,
                         'quaternion': hp.quaternion.tolist(),
                         'label': self.visible_labels[i]}
                    )
                self.rotation_map.add_quaternion(
                    int(px), int(py), hp.quaternion)

                # self.depth_map.add_depth(
                #     int(px), int(py),
                #     hp.worldpos()[2] * 1000)

        if np.all(self.annotation_img == 0):
            print('out of camera')
            return False

        self.annotation_img \
            = self.annotation_img / self.annotation_img.max() * 255
        self.annotation_img = self.annotation_img.astype(np.uint8)

        self.rotations = self.rotation_map.rotations

        # self.hanging_points_depth = self.depth_map.on_depth_image(self.depth)

        return True

    def get_data_id(self):
        """Get the number of saved data

        Returns
        -------
        self.data_id : int
        """
        self.data_id = len(glob.glob(osp.join(self.save_dir, 'depth', '*npy')))
        return self.data_id

    def save_data(self):
        """Save training data"""
        print('Save {}'.format(self.data_id))
        cv2.imwrite(osp.join(self.save_dir, 'color', '{:06}.png'.format(
            self.data_id)), self.bgr)
        np.save(osp.join(
            self.save_dir, 'depth', '{:06}'.format(self.data_id)), self.depth)
        # np.save(osp.join(
        #     self.save_dir, 'hanging_points_depth', '{:06}'.format(
        #         self.data_id)), self.hanging_points_depth)
        # np.save(osp.join(
        #     self.save_dir, 'rotations', '{:06}'.format(
        #         self.data_id)), self.rotations)
        save_json(osp.join(self.save_dir, 'annotation',
                           '{:06}.json'.format(self.data_id)),
                  self.annotation_data)
        # cv2.imwrite(osp.join(
        #     self.save_dir, 'heatmap', '{:06}.png'.format(self.data_id)),
        #     self.annotation_img)
        self.camera_model.dump(osp.join(
            self.save_dir, 'camera_info', '{:06}.yaml'.format(self.data_id)))

        if self.save_debug_image:
            cv2.imwrite(osp.join(
                self.save_dir, 'debug_axis', '{:06}.png'.format(self.data_id)),
                self.bgr_axis)
            cv2.imwrite(osp.join(
                self.save_dir, 'debug_roi', '{:06}.png'.format(self.data_id)),
                self.roi_image)
            _object_mask = np.zeros_like(self.roi_image)
            _object_mask[self.object_mask] = 255  # idx -> image
            cv2.imwrite(osp.join(
                self.save_dir, 'debug_mask', '{:06}.png'.format(self.data_id)),
                _object_mask)

    def create_data(self, urdf_file, contact_points,
                    random_pose=True, random_texture=True,
                    use_align_coords=False, use_average_coords=False,
                    translate=[0, 0.01, 0]):
        """Create training data

        Parameters
        ----------
        urdf_file : str
        contact_points : list[list[list[float], list[float]]]
        translate : list, optional
            translate contact points for ray-trace,
            by default [0, 0.01, 0]. this is for hanging.

        Returns
        -------
        result : bool
        """
        self.get_data_id()
        self.get_plane()
        self.load_urdf(
            urdf_file,
            random_pose=random_pose,
            random_texture=random_texture)
        contact_points_coords = self.make_contact_points_coords(contact_points)

        if use_align_coords:
            contact_points_coords \
                = self.align_coords(
                    contact_points_coords, copy_list=False)
            if use_average_coords:
                contact_points_coords \
                    = self.make_average_coords_list(contact_points_coords)

        if random_texture:
            self.change_texture(self.plane_id)
        if random_texture:
            self.change_texture(self.object_id)
        self.create_camera()
        loop = True

        self.no_visible_count = 0
        while loop:
            self.move_to_random_pos()
            self.look_at(self.object_coords.worldpos()
                         - self.object_center
                         + (np.random.rand(3) - 0.5) * 0.2)
            self.step(1)
            print('self.no_visible_count %d' % self.no_visible_count)
            if self.no_visible_count >= self.no_visible_skip_num:
                return False
            if not self.get_visible_coords(
                    contact_points_coords,
                    translate=self.translate_value):
                self.reset_object_pose()
                self.no_visible_count += 1
                continue
            else:
                loop = False
            self.no_visible_count = 0

            self.get_bgr()
            self.get_seg()

            if self.get_object_mask(self.object_id) is None:
                self.reset_object_pose()
                continue

            self.get_object_depth()

            self.crop(padding=50)

        if not self.create_annotation_data():
            self.finish()
            self.remove_all_objects()
            return False
        self.save_data()

        if self.stop_per_data:
            input('Next data?: [ENTER]')

        self.finish()
        self.remove_all_objects()
        return True

    def create_eval_data(
            self, urdf_file, contact_points, camera_pos, obj_pos, obj_rot,
            random_pose=False, random_texture=False):
        """Create ycb eval data

        Parameters
        ----------
        urdf_file : str
        contact_points : list[list[list[float], list[float]]]

        Returns
        -------
        result : bool
        """
        self.get_data_id()
        self.get_plane()
        self.load_urdf(
            urdf_file,
            random_pose=random_pose,
            random_texture=random_texture,
            pos=obj_pos,
            rot=obj_rot)
        contact_points_coords = self.make_contact_points_coords(contact_points)
        if random_texture:
            self.change_texture(self.plane_id)
        if random_texture:
            self.change_texture(self.object_id)
        self.create_camera()

        self.no_visible_count = 0
        self.move_to(camera_pos)
        self.look_at(self.object_coords.worldpos()
                     - self.object_center + [0, 0, 0.1], horizontal=True)
        self.step(1)
        print('self.no_visible_count %d' % self.no_visible_count)
        if self.no_visible_count >= self.no_visible_skip_num:
            self.finish()
            return False
        if not self.get_visible_coords(
                contact_points_coords,
                translate=self.translate_value,
                all_visible=True):
            self.no_visible_count += 1
            self.finish()
            return False

        self.no_visible_count = 0

        bgr = self.get_bgr()
        cv2.imshow('bgr', bgr)
        if cv2.waitKey(0) != ord('y'):
            print('skip image')
            cv2.destroyAllWindows()
            self.finish()
            return False
        print('save image')
        cv2.destroyAllWindows()

        self.get_seg()

        if self.get_object_mask(self.object_id) is None:
            self.finish()
            return False

        self.get_object_depth()

        self.crop(padding=50)

        if not self.create_annotation_data():
            self.finish()
            return False

        self.save_data()
        self.finish()
        return True

    def from_camera_pose(self, camera_pose_path):
        """Load camera pose file and set camera coords

        Parameters
        ----------
        camera_pose_path : str
        """
        coords = load_camera_pose(camera_pose_path)
        self.move_to_coords(coords)

    def get_sim_images(self, urdf_file, camera_pose_path):
        """Get simulation images

        Used to obtain a simulation image from a mesh generated from real data.

        Parameters
        ----------
        urdf_file : str
        camera_pose_path : str

        Returns
        -------
        self.bgr, self.depth
        """
        self.load_urdf(urdf_file, random_pose=False)
        # self.get_plane()
        # self.change_texture(self.plane_id)
        # self.change_texture(self.object_id)

        self.create_camera()
        self.from_camera_pose(camera_pose_path)
        self.step(1)

        self.get_bgr()
        self.get_seg()

        if self.get_object_mask(self.object_id) is None:
            return False

        self.get_object_depth()
        self.crop(padding=10, random=False)

        print('sim img')

        return self.bgr, self.depth

    def finish(self):
        """Finish simulation"""
        self.remove_all_objects()
        pybullet.resetSimulation()
        pybullet.disconnect()


class DepthMap():
    def __init__(self, width, height, circular=True):
        """Depth map which store annotated depth value

        Parameters
        ----------
        width : int
        height : int
        circular : bool, optional
            Annotate the map in a circle
        """
        self.width = width
        self.height = height
        self.size = width * height
        self.idx_list = []
        self._depth_buffer = [[] for _ in range(self.size)]
        self._depth = [0] * self.size
        self.circular = circular

    def add_depth(self, px, py, d):
        """Add depth to target pixel

        Parameters
        ----------
        px : int
        py : int
        d : float
            depth value
        """
        if self.circular:
            iy, ix = np.where(
                create_circular_mask(
                    self.height, self.width, py, px, radius=50))
            idices = ix + iy * self.width
            for idx in idices:
                self._depth_buffer[idx].append(d)
        else:
            idx = px + py * self.width
            self._depth[idx] = d

    def calc_average_depth(self):
        """Calculate average depth each pixel"""
        for idx in range(self.size):
            if self._depth_buffer[idx] != []:
                self._depth[idx] = np.mean(self._depth_buffer[idx])

    @property
    def depth(self):
        """Depth

        Returns
        -------
        depth : numpy.ndarray
        """
        if self.circular:
            self.calc_average_depth()
        return np.array(self._depth).reshape(self.height, self.width)

    def on_depth_image(self, depth_image):
        """Overlay map depth on input depth

        Parameters
        ----------
        depth_image : numpy.ndarray

        Returns
        -------
        depth_image ; numpy.ndarray
        """
        depth_image = depth_image.copy()
        mask = np.where(self.depth != 0)
        depth_image[mask] = self.depth[mask]
        return depth_image


class RotationMap():
    def __init__(self, width, height, average=False):
        """Rotation map which store annotated quatenion

        Parameters
        ----------
        width : int
        height : int
        """
        self.width = width
        self.height = height
        self.size = width * height
        self.average = average
        self._rotations_buffer = [[] for _ in range(self.size)]
        # [w, x, y, z] quaternion
        self._rotations = [np.array([1, 0, 0, 0])] * self.size

    def add_quaternion(self, px, py, q):
        """Add quaternion to target pixel

        Parameters
        ----------
        px : int
        py : int
        q : list[flaot]
            quaternion
        """
        iy, ix = np.where(
            create_circular_mask(self.height, self.width, py, px, radius=50))
        idices = ix + iy * self.width
        for idx in idices:
            if self.average:
                self._rotations_buffer[idx].append(q.tolist())
            self._rotations[idx] = q.tolist()

    def get_length(self, px, py):
        """Get the number of stored quaternion in target pixel

        Parameters
        ----------
        px : int
        py : int

        Returns
        -------
        len(self.rotations_buffer[idx]) : int
        """
        idx = px + py * self.width
        return len(self._rotations_buffer[idx])

    def calc_average_rotations(self):
        """Calculate average quaternion"""
        for idx in range(self.size):
            if self._rotations_buffer[idx] != []:
                self._rotations[idx] = averageQuaternions(
                    np.array(self._rotations_buffer[idx]))

    @property
    def rotations(self):
        if self.average:
            self.calc_average_rotations()
        return np.array(self._rotations).reshape(self.height, self.width, 4)


def get_contact_points(contact_points_path, json_name='contact_points.json',
                       dataset_type='ycb', use_clustering=True,
                       use_filter_penetration=True,
                       inf_penetration_check=True, get_label=False):
    """Get contact points from file

    Parameters
    ----------
    contact_points_path : str
        dir or file path.
        if dir load multiple file.
    json_name : str, optional
        target json file name, by default 'contact_points.json'
    dataset_type : str, optional
        by default 'ycb'
    use_clustering : bool, optional
        by default True
    use_filter_penetration : bool, optional
        by default True
    inf_penetration_check : bool, optional
        by default True
    get_label : bool, optional
        by default True
        This only works if use_clustering, use_filter_penetration
        and inf_penetration_check are all false.

    Returns
    -------
    contact_points : list[list[list[float], list[float]]]
    """

    if osp.isdir(contact_points_path):
        contact_points_dict = load_multiple_contact_points(
            contact_points_path, json_name)
    else:
        contact_points_dict = json.load(open(contact_points_path, 'r'))
    contact_points = contact_points_dict['contact_points']



    if use_clustering:
        contact_points = cluster_contact_points(
            contact_points, min_samples=-1)
        get_label = False

    if use_filter_penetration:
        if inf_penetration_check:
            box_size = [100, 0.0001, 0.0001]
        else:
            box_size = [0.1, 0.0001, 0.0001]

        dirname = osp.dirname(contact_points_path)
        if dataset_type == 'ycb':
            contact_points, _ = filter_penetration(
                osp.join(dirname, 'base.urdf'),
                contact_points, box_size=box_size)
        else:
            contact_points, _ = filter_penetration(
                osp.join(dirname, urdf_name),
                contact_points, box_size=box_size)
        get_label = False

    if len(contact_points) == 0:
        print('num of hanging points: {}'.format(len(contact_points)))
        return None

    if get_label:
        labels = contact_points_dict['labels']
        return contact_points, labels
    else:
        return contact_points


def sample_contact_points(contact_points, num_samples, get_idx=False):
    """Sampling contact points for the specified number

    Parameters
    ----------
    contact_points : list[list[list[float], list[float]]]
    num_samples : int

    Returns
    -------
    contact_points : list[list[list[float], list[float]]]
    """
    num_samples = min(len(contact_points), num_samples)
    idx = random.sample(range(0, len(contact_points)), num_samples)
    if get_idx:
        return [contact_points[i] for i in idx], idx
    else:
        return [contact_points[i] for i in idx]


def make_save_dirs(save_dir, debug_dir=True):
    """Make each save dir

    Parameters
    ----------
    save_dir : str
        base save dir

    Returns
    -------
    save_dir ; str
        save dir made by eos.make_fancy_output_dir
    """
    save_dir = make_fancy_output_dir(save_dir,
                                     save_environ=False, save_command=False,
                                     save_git=False, save_gitignore=False,
                                     save_pip=False)
    os.makedirs(osp.join(save_dir, 'color'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'depth'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'camera_info'), exist_ok=True)
    os.makedirs(osp.join(save_dir, 'annotation'), exist_ok=True)

    if debug_dir:
        os.makedirs(osp.join(save_dir, 'debug_axis'), exist_ok=True)
        os.makedirs(osp.join(save_dir, 'debug_roi'), exist_ok=True)
        os.makedirs(osp.join(save_dir, 'debug_mask'), exist_ok=True)
    return save_dir


def split_file_name(file, dataset_type='ycb'):
    """Split object file name and get dirname, filename, category_name, idx

    Parameters
    ----------
    file : str
    dataset_type : str, optional
        'ycb' or 'ObjectNet3D', by default 'ycb'

    Returns
    -------
    dirname : str
    filename : str
    category_name : str
    idx : int
        if only 'ObjectNet3D'.
        One type of object has multiple ids
    """
    dirname, filename = osp.split(file)
    filename_without_ext, ext = osp.splitext(filename)

    if dataset_type == 'ObjectNet3D':
        category_name = dirname.split("/")[-2]
        idx = dirname.split("/")[-1]
    else:  # ycb
        category_name = dirname.split("/")[-1]
        idx = None
    return dirname, filename, category_name, idx


def averageQuaternions(Q):
    """Calculate average quaternion

    https://github.com/christophhagen/averaging-quaternions/blob/master/LICENSE
    Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
    The quaternions are arranged as (w,x,y,z), with w being the scalar
    The result will be the average quaternion of the input. Note that the signs
    of the output quaternion can be reversed, since q and -q describe the same orientation

    Parameters
    ----------
    Q : numpy.ndarray or list[float]

    Returns
    -------
    average quaternion : numpy.ndarray
    """

    Q = np.array(Q)
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


def average_coords(coords_list):
    """Caluc average coords

    Parameters
    ----------
    coords_list : list[skrobot.coordinates.Coordinates]

    Returns
    -------
    coords_average : skrobot.coordinates.Coordinates
    """
    q_list = [c.quaternion for c in coords_list]
    q_average = averageQuaternions(q_list)
    pos_average = np.mean([c.worldpos() for c in coords_list], axis=0)
    coords_average = coordinates.Coordinates(pos_average, q_average)
    return coords_average


if __name__ == '__main__':
    print('Start')
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--save-dir', '-s',
        type=str, help='save dir',
        default='/media/kosuke/SANDISK-2/meshdata/hanging_object/mthesi_media')
    parser.add_argument(
        '--save-debug-image', '-sdd',
        action='store_true', help='If true, save debug image')
    parser.add_argument(
        '--data-num', '-n',
        type=int, help='num of data per object',
        default=1000)
    parser.add_argument(
        '--num-samples', '-ns',
        type=int, help='number of sampled data for rendering',
        default=None)
    parser.add_argument(
        '--input-dir', '-i',
        type=str, help='input dir',
        # default='/media/kosuke/SANDISK/meshdata/hanging_object')
        # default='/media/kosuke55/SANDISK/meshdata/random_shape_shapenet_hanging_1016')
        default='/media/kosuke55/SANDISK/meshdata/shapenet_mini_pouring_50')
    parser.add_argument(
        '--dataset-type', '-dt',
        type=str, help='dataset type',
        default='shapenet')
    parser.add_argument(
        '--urdf-name', '-u',
        type=str, help='urdf file name',
        default='base.urdf')
    parser.add_argument(
        '--skip-list', '-sl',
        type=str, help='skip list file name',
        default='filter_skip_list.txt')
    parser.add_argument(
        '--remain-list', '-rl',
        type=str, help='remain list file name',
        default=None)
    parser.add_argument(
        '--gui', '-g',
        action='store_true', help='debug gui')
    parser.add_argument(
        '--show-image', '-si',
        action='store_true', help='show image')
    parser.add_argument(
        '--filtered-points-name', '-fjn',
        type=str, help='filtered points json file name',
        default='filtered_contact_points.json')
        # default='filtered_contact_points_pouring.json')
    parser.add_argument(
        '--task-type', '-t', type=str,
        default=None,
        help='if None, set it automaticaly from outfile')
    parser.add_argument(
        '--stop-per-data', '-spd',
        action='store_true',
        help='stop the window per data for checking')
    args = parser.parse_args()

    data_num = args.data_num
    dataset_type = args.dataset_type
    gui = args.gui
    input_dir = args.input_dir
    save_dir_base = args.save_dir
    urdf_name = args.urdf_name
    show_image = args.show_image
    save_debug_image = args.save_debug_image
    filtered_points_name = args.filtered_points_name
    stop_per_data = args.stop_per_data

    if args.task_type is not None:
        task_type = args.task_type
    else:
        if 'hanging' in input_dir:
            task_type = 'hanging'
        elif 'pouring' in input_dir:
            task_type = 'pouring'
    print('task_type ', task_type)

    file_paths = list(sorted(Path(
        input_dir).glob(osp.join('*', urdf_name))))
    files = list(map(str, file_paths))

    if dataset_type == 'shapenet':
        if task_type == 'hanging':
            category_name_list = hanging_label()
        elif task_type == 'pouring':
            category_name_list = pouring_label()

    elif dataset_type == 'ycb':
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

    skip_list = []
    if args.skip_list is not None:
        skip_list = args.skip_list
        skip_list_file = str(Path(input_dir) / skip_list)
        if osp.isfile(skip_list_file):
            skip_list = load_list(skip_list_file)

    if args.remain_list is not None:
        remain_list = args.remain_list
        remain_list_file = str(Path(input_dir) / remain_list)
        remained_objects = load_list(remain_list_file)
        remained_files = []
        for file in files:
            dirname, filename, category_name, idx \
                = split_file_name(file, dataset_type)
            if category_name in remained_objects:
                remained_files.append(file)
        files = remained_files
        print('Load file from ' + remain_list_file)
    else:
        remained_files = []
        for file in files:
            dirname, filename, category_name, idx \
                = split_file_name(file, dataset_type)
            if category_name in skip_list:
                print('skip {} (in skip_list)'.format(file))
                continue
            if not any(c in category_name for c in category_name_list):
                print('skip {} (not in category list)'.format(file))
                continue
            remained_files.append(file)
        files = remained_files

    if args.num_samples is not None:
        num_samples = args.num_samples
        indices = random.sample(range(0, len(files)), num_samples)
        files = [files[i] for i in indices]
        print('samping {} files'.format(len(files)))

    print('number of files {}'.format(len(files)))

    try:
        for file in files:
            dirname, filename, category_name, idx \
                = split_file_name(file, dataset_type)

            if not osp.isfile(
                    osp.join(dirname, filtered_points_name)):
                print('Skipped %s because no ' % category_name
                      + filtered_points_name)
                continue
            contact_points = get_contact_points(
                osp.join(dirname, filtered_points_name),
                use_clustering=False, use_filter_penetration=False,
                inf_penetration_check=False)

            if contact_points is None:
                continue

            save_dir = osp.join(save_dir_base, category_name)
            save_dir = make_save_dirs(
                save_dir, debug_dir=save_debug_image)

            contact_points = sample_contact_points(contact_points, 30)
            while True:
                r = Renderer(
                    gui=gui,
                    save_dir=save_dir,
                    save_debug_image=save_debug_image,
                    task_type=task_type,
                    stop_per_data=stop_per_data)
                r.create_data(osp.join(dirname, urdf_name), contact_points)
                if r.no_visible_count >= r.no_visible_skip_num:
                    print('Skip because this object has no visible points ')
                    shutil.rmtree(save_dir)
                    break

                r.get_data_id()
                print(r.data_id)
                if r.data_id == data_num:
                    break

    except KeyboardInterrupt:
        sys.exit()
