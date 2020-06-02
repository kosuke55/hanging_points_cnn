#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import glob
import json
import numpy.matlib as npm
import os
import sys

import cameramodels
import numpy as np
import pybullet
import pybullet_data
# import skrobot

import xml.etree.ElementTree as ET
from hanging_points_generator.hanging_points_generator \
    import cluster_hanging_points
from sklearn.cluster import DBSCAN
from skrobot import coordinates


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


def create_circular_mask(h, w, cy, cx, radius=50):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = dist_from_center <= radius
    return mask


def create_depth_circle(img, cy, cx, value, radius=50):
    depth_mask = np.zeros_like(img)
    depth_mask[np.where(img == 0)] = 1
    # depth_mask = np.where(img == 0)
    # print(depth_mask)
    circlular_mask = np.zeros_like(img)
    circlular_mask_idx = np.where(
        create_circular_mask(img.shape[0], img.shape[1], cy, cx,
                             radius=radius))
    circlular_mask[circlular_mask_idx] = 1

    # mask = np.where(np.logical_and(depth_mask == 1, circlular_mask == 1))
    # circlular_mask = np.where(
    #     create_circular_mask(img.shape[0], img.shape[1], cy, cx))
    # mask = np.logical_and(depth_mask, circlular_mask)

    # img[mask] = value
    img[circlular_mask_idx] = value


def create_gradient_circle(img, cy, cx, sig=50., gain=100.):
    h, w = img.shape
    Y, X = np.ogrid[:h, :w]
    g = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2. * sig)) * gain
    img += g.astype(np.uint32)


def colorize_depth(depth, min_value=None, max_value=None):
    min_value = np.nanmin(depth) if min_value is None else min_value
    max_value = np.nanmax(depth) if max_value is None else max_value

    gray_depth = depth.copy()
    nan_mask = np.isnan(gray_depth)
    gray_depth[nan_mask] = 0
    gray_depth = 255 * (gray_depth - min_value) / (max_value - min_value)
    gray_depth[gray_depth <= 0] = 0
    gray_depth[gray_depth > 255] = 255
    gray_depth = gray_depth.astype(np.uint8)
    colorized = cv2.applyColorMap(gray_depth, cv2.COLORMAP_JET)
    colorized[nan_mask] = (0, 0, 0)

    return colorized


def draw_axis(img, R, t, K):
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32(
        [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(
        img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()),
        (0, 0, 255), 1)
    img = cv2.line(
        img,
        tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()),
        (0, 255, 0), 1)
    img = cv2.line(
        img,
        tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()),
        (255, 0, 0), 1)
    return img


class Renderer:
    def __init__(
            self,
            im_width,
            im_height,
            fov,
            near_plane,
            far_plane,
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
        self.pm = pybullet.computeProjectionMatrixFOV(
            fov, aspect, near_plane, far_plane)

        self.camera_coords = coordinates.Coordinates(
            pos=np.array([0, 0, 0.5]),
            rot=coordinates.math.rotation_matrix_from_rpy([0, np.pi, 0]))

        # self.object_pos = np.array([0, 0, 0.1])
        # self.object_rot = coordinates.math.rotation_matrix_from_rpy([0, 0, 0])

        self.object_coords = coordinates.Coordinates(
            pos=np.array([0, 0, 0.1]),
            rot=coordinates.math.rotation_matrix_from_rpy([0, 0, 0]))

        if DEBUG:
            self.cid = pybullet.connect(pybullet.GUI)
        else:
            self.cid = pybullet.connect(pybullet.DIRECT)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.draw_camera_pos()
        self._rendered = None
        self._rendered_pos = None
        self._rendered_rot = None

    def load_urdf(self, urdf):
        object_id = pybullet.loadURDF(urdf,
                                      [0, 0, 0.1], [0, 0, 0, 1])
        self.objects.append(object_id)
        return object_id

    def remove_object(self, o_id, update=True):
        pybullet.removeBody(o_id)
        if update:
            # print("remove ", o_id)
            self.objects.remove(o_id)

    def remove_all_objects(self):
        objects = copy.copy(self.objects)
        for o_id in objects:
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

    def render(self):
        if np.all(
                self._rendered_pos == self.camera_coords.worldpos()) and np.all(
                self._rendered_rot == self.camera_coords.worldrot()):
            return self._rendered

        target = self.camera_coords.worldpos() + \
            self.camera_coords.rotate_vector([0, 0, 1.])
        up = self.camera_coords.rotate_vector([0, -1.0, 0])
        # up = self.camera_coords.copy().rotate_vector([0, 1.0, 0])

        vm = pybullet.computeViewMatrix(
            self.camera_coords.worldpos(), target, up)

        i_arr = pybullet.getCameraImage(
            self.im_width, self.im_height, vm, self.pm,
            shadow=0,
            renderer=pybullet.ER_TINY_RENDERER)

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

    def get_rgb(self):
        return self.render()[2]

    def get_seg(self):
        return self.render()[4]

    def move_to(self, T):
        self.camera_coords = coordinates.Coordinates(
            pos=np.array(T),
            rot=r.camera_coords.worldrot())

    def look_at(self, p):
        p = np.array(p)
        if np.all(p == self.camera_coords.worldpos()):
            return
        z = p - self.camera_coords.worldpos()
        coordinates.geo.orient_coords_to_axis(self.camera_coords, z)

        self.draw_camera_pos()


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


if __name__ == '__main__':
    print('Start')
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--save_dir',
        '-s',
        type=str,
        help='save dir',
        default='ycb_hanging_object/0603')
    parser.add_argument(
        '--input_files',
        '-i',
        type=str,
        help='input files',
        default='ycb_hanging_object/urdf/*/*')
    parser.add_argument(
        '--urdf_name',
        '-u',
        type=str,
        help='save dir',
        default='textured.urdf')
    args = parser.parse_args()

    save_dir = args.save_dir
    urdf_name = args.urdf_name
    files = glob.glob(args.input_files)

    im_w = 1920
    im_h = 1080
    im_fov = 42.5
    nf = 0.1
    ff = 2.0
    width = 256
    height = 256

    # save_dir = 'Hanging-ObjectNet3D-DoubleFaces/cup_key_scissors_0528'
    # save_dir = 'ycb_hanging_object/0601'
    # files = glob.glob("Hanging-ObjectNet3D-DoubleFaces/CAD.selected/urdf/*/*/*")
    # files = glob.glob("ycb_hanging_object/urdf/*/*")
    # urdf_name = 'textured.urdf'

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'intrinsics'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'color'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'color_raw'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'debug_axis'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'debug_heatmap'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'depth_bgr'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'hanging_points_depth'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'heatmap'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'rotations'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'clip_info'), exist_ok=True)

    if 'ycb' in args.input_files:
        category_name_list = [
            "019_pitcher_base",
            "022_windex_bottle",
            "025_mug",
            "033_spatula",
            "035_power_drill",
            "042_adjustable_wrench",
            "048_hammer",
            "050_medium_clamp",
            "051_large_clamp",
            "052_extra_large_clamp"
        ]
    elif 'ObjectNet3D' in args.input_files:
        category_name_list = ['cup', 'key', 'scissors']

    # r = Renderer(im_w, im_h, im_fov, nf, ff, DEBUG=True)
    r = Renderer(im_w, im_h, im_fov, nf, ff, DEBUG=False)
    np.save(
        os.path.join(
            save_dir, 'intrinsics', 'intrinsics'), r.camera_model.K)

    camera_length = 0.01
    camera_object = pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=[camera_length, camera_length, camera_length])
    # r = Renderer(im_w, im_h, im_fov, nf, ff, DEBUG=False)

    print(files)
    data_id = 0
    try:
        for file in files:
            dirname, filename = os.path.split(file)
            filename_without_ext, ext = os.path.splitext(filename)
            if 'ycb' in args.input_files:
                category_name = dirname.split("/")[-1]
            elif 'ObjectNet3D' in args.input_files:
                category_name = dirname.split("/")[-2]
                idx = dirname.split("/")[-1]
            print(category_name)
            # if category_name not in category_name_list:
            # continue
            if category_name == "033_spatula":
                continue
            # indices = ['01', '02', '03']
            # indices = ['01']
            # if idx not in indices:
            #     continue
            if filename == urdf_name:
                tree = ET.parse(os.path.join(dirname, urdf_name))
                root = tree.getroot()

                center = np.array([float(i) for i in root[0].find(
                    "inertial").find("origin").attrib['xyz'].split(' ')])

                contact_points_dict = json.load(
                    open(os.path.join(dirname, 'contact_points.json'), 'r'))
                contact_points = contact_points_dict['contact_points']

                contact_points = cluster_hanging_points(
                    contact_points, eps=0.005, min_samples=2)

                r.step(1)
                r.look_at([0, 0, 2])

                data_count = 0
                # for _ in range(1000):
                while data_count < 100:
                    print('{}: {} sum: {}'.format(file, data_count, data_id))
                    camera_id = pybullet.createMultiBody(
                        baseMass=0.,
                        baseCollisionShapeIndex=camera_object,
                        basePosition=r.camera_coords.worldpos(),
                        baseOrientation=[0, 0, 0, 1.])
                    r.objects.append(camera_id)

                    # tree = ET.parse(os.path.join(dirname, urdf_name))
                    # root = tree.getroot()
                    # # mesh_scale_list = [(np.random.rand() - 0.5) * 0.5 + 1,
                    # #                    (np.random.rand() - 0.5) * 0.5 + 1,
                    # #                    (np.random.rand() - 0.5) * 0.5 + 1]
                    mesh_scale_list = [1, 1, 1]
                    # mesh_scale = ''.join(str(i) + ' ' for i in mesh_scale_list).strip()
                    # root[0].find('visual').find('geometry').find('mesh').attrib['scale'] = mesh_scale
                    # root[0].find('collision').find('geometry').find('mesh').attrib['scale'] = mesh_scale
                    # tree.write(os.path.join(dirname, 'rescale_base.urdf'),
                    #            encoding='utf-8', xml_declaration=True)
                    # object_id = r.load_urdf(os.path.join(dirname, "rescale_base.urdf"))

                    # color = np.random.rand(3).tolist()
                    # color.append(1)
                    # pybullet.changeVisualShape(object_id, -1, rgbaColor=color)

                    object_id = r.load_urdf(os.path.join(dirname, urdf_name))
                    newpos = [0, 0, 0]
                    while np.linalg.norm(newpos) < 0.3:
                        newpos = [(np.random.rand() - 0.5),
                                  (np.random.rand() - 0.5),
                                  (np.random.rand() - 0.5)]

                    r.move_to(newpos)

                    pos, rot \
                        = pybullet.getBasePositionAndOrientation(object_id)
                    r.object_coords = coordinates.Coordinates(
                        pos=pos,
                        rot=coordinates.math.xyzw2wxyz(rot))

                    r.camera_coords = coordinates.Coordinates(
                        pos=r.camera_coords.worldpos(),
                        rot=coordinates.math.matrix2quaternion(
                            r.camera_coords.worldrot()))

                    r.look_at(r.object_coords.worldpos() - center)

                    pybullet.resetBasePositionAndOrientation(
                        camera_id,
                        r.camera_coords.worldpos(),
                        coordinates.math.wxyz2xyzw(
                            coordinates.math.matrix2quaternion(
                                r.camera_coords.worldrot())))

                    r.step(1)

                    depth = (r.get_depth_metres() * 1000).astype(np.float32)

                    rgb = r.get_rgb()
                    # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    bgr_axis = bgr.copy()

                    seg = r.get_seg()

                    if np.count_nonzero(seg == object_id) == 0:
                        print("continue")
                        r.remove_all_objects()
                        continue
                    object_mask = np.where(seg == object_id)
                    non_object_mask = np.where(seg != object_id)
                    depth[non_object_mask] = 0
                    bgr[non_object_mask] = [0, 0, 0]

                    annotation_img = np.zeros_like(seg, dtype=np.uint8)

                    ymin = np.min(object_mask[0])
                    ymax = np.max(object_mask[0])
                    xmin = np.min(object_mask[1])
                    xmax = np.max(object_mask[1])

                    bgr_raw = bgr.copy()
                    bgr = bgr[ymin:ymax, xmin:xmax]

                    scale = float(width) / np.array(bgr.shape[:2])

                    bgr = cv2.resize(bgr, (width, height))

                    depth_bgr = colorize_depth(depth, 100, 1500)
                    depth_bgr = depth_bgr[ymin:ymax, xmin:xmax]
                    depth_bgr = cv2.resize(depth_bgr, (width, height))

                    # depth = depth[ymin:ymax, xmin:xmax]
                    # depth = cv2.resize(depth, (width, height))

                    # depth = depth[ymin:ymax, xmin:xmax]
                    # depth = cv2.resize(depth, (width, height))
                    # print(np.min(depth))
                    # print(np.mean(depth))
                    # print(np.max(depth))

                    annotation_img = annotation_img[ymin:ymax, xmin:xmax]
                    annotation_img = cv2.resize(
                        annotation_img, (width, height))
                    annotation_img = annotation_img.astype(np.uint32)
                    rotation_map = RotationMap(width, height)

                    hanging_point_in_camera_coords_list = []

                    for cp in contact_points:
                        hanging_point_coords = coordinates.Coordinates(
                            pos=(cp[0] - center) * mesh_scale_list, rot=cp[1:])

                        # hanging_point_coords.translate([0, 0.01, 0])

                        hanging_point_worldcoords \
                            = r.object_coords.copy().transform(
                                hanging_point_coords)
                        pybullet.addUserDebugLine(
                            hanging_point_worldcoords.worldpos(),
                            r.camera_coords.worldpos(), [1, 1, 1], 2)

                        hanging_point_in_camera_coords \
                            = r.camera_coords.inverse_transformation(
                            ).transform(hanging_point_worldcoords)

                        px, py = r.camera_model.project3d_to_pixel(
                            hanging_point_in_camera_coords.worldpos())

                        rayInfo = pybullet.rayTest(
                            hanging_point_worldcoords.worldpos(),
                            r.camera_coords.worldpos())

                        if rayInfo[0][0] == camera_id:
                            hanging_point_in_camera_coords_list.append(
                                hanging_point_in_camera_coords)

                    # for hp in hanging_point_in_camera_coords_list:
                    #     R = hp.worldrot()
                    #     t = hp.worldpos()
                    #     draw_axis(bgr_axis, R, t, r.camera_model.K)

                    if len(hanging_point_in_camera_coords_list) == 0:
                        r.remove_all_objects()
                        continue
                    dbscan = DBSCAN(
                        eps=0.005, min_samples=2).fit(
                            [hp.worldpos() for hp in
                             hanging_point_in_camera_coords_list])

                    quaternion_list = []

                    # hanging_points_depth = np.zeros_like(depth)
                    hanging_points_depth = depth.copy()

                    for label in range(np.max(dbscan.labels_) + 1):
                        if np.count_nonzero(dbscan.labels_ == label) <= 1:
                            # print("skip label ", label)
                            continue
                        q_base = None
                        for idx, hp in enumerate(
                                hanging_point_in_camera_coords_list):
                            if dbscan.labels_[idx] == label:
                                if q_base is None:
                                    q_base = hp.quaternion
                                q_distance \
                                    = coordinates.math.quaternion_distance(
                                        q_base, hp.quaternion)
                                # print(label, idx, np.rad2deg(q_distance))
                                if np.rad2deg(q_distance) > 135:
                                    hanging_point_in_camera_coords_list[
                                        idx].rotate(np.pi, 'y')

                                px, py = r.camera_model.project3d_to_pixel(
                                    hp.worldpos())

                                draw_axis(bgr_axis,
                                          hp.worldrot(),
                                          hp.worldpos(),
                                          r.camera_model.K)

                                if int(scale[1] * (-xmin + px)) >= 0 and \
                                   int(scale[1] * (-xmin + px)) < width and \
                                   int(scale[0] * (-ymin + py)) >= 0 and \
                                   int(scale[0] * (-ymin + py)) < height:

                                    # gradient_circle(
                                    #     annotation_img,
                                    #     int(scale[0] * (-ymin + py)),
                                    #     int(scale[1] * (-xmin + px)))

                                    create_gradient_circle(
                                        annotation_img,
                                        int(scale[0] * (-ymin + py)),
                                        int(scale[1] * (-xmin + px)))

                                    rotation_map.add_quaternion(
                                        int(scale[1] * (-xmin + px)),
                                        int(scale[0] * (-ymin + py)),
                                        hanging_point_in_camera_coords_list[
                                            idx].quaternion)

                                    # create_depth_circle(
                                    #     hanging_points_depth,
                                    #     int(scale[0] * (-ymin + py)),
                                    #     int(scale[1] * (-xmin + px)),
                                    #     hp.worldpos()[2] * 1000)

                                    create_depth_circle(
                                        hanging_points_depth,
                                        py,
                                        px,
                                        hp.worldpos()[2] * 1000)

                    annotation_img[np.where(annotation_img >= 256)] = 255
                    annotation_img = annotation_img.astype(np.uint8)
                    annotation_color = np.zeros_like(bgr, dtype=np.uint8)
                    annotation_color[..., 2] = annotation_img
                    bgr_annotation = cv2.addWeighted(bgr, 0.3,
                                                     annotation_color, 0.7, 0)

                    # rotation_map.calc_average_rotations(annotation_img)
                    rotation_map.calc_average_rotations()
                    rotations = np.array(
                        rotation_map.rotations).reshape(height, width, 4)

                    # print(rotations[np.where(
                    #     np.any(rotations != [0, 0, 0, 1], axis=2))])

                    bgr_axis = bgr_axis[ymin:ymax, xmin:xmax]
                    bgr_axis = cv2.resize(bgr_axis, (width, height))

                    depth = depth[ymin:ymax, xmin:xmax]
                    depth = cv2.resize(depth, (width, height))

                    hanging_points_depth_bgr \
                        = colorize_depth(hanging_points_depth, 100, 1500)
                    hanging_points_depth_bgr = hanging_points_depth_bgr[
                        ymin:ymax, xmin:xmax]
                    hanging_points_depth_bgr \
                        = cv2.resize(hanging_points_depth_bgr, (width, height))

                    hanging_points_depth = hanging_points_depth[
                        ymin:ymax, xmin:xmax]
                    hanging_points_depth \
                        = cv2.resize(hanging_points_depth, (width, height))

                    # cv2.imshow('annotation', annotation_img)
                    # cv2.imshow('bgr', bgr_annotation)
                    # cv2.imshow('bgr_axis', bgr_axis)
                    # cv2.imshow('depth_bgr', depth_bgr)
                    # cv2.imshow('hanging_points_depth_bgr',
                    #            hanging_points_depth_bgr)
                    # cv2.imshow('depth', depth)
                    # cv2.imshow('hanging_points_depth', hanging_points_depth)
                    # cv2.waitKey(1)

                    clip_info = np.array([xmin, xmax, ymin, ymax])

                    # if data_id == 0:
                    #     cv2.moveWindow('bgr', 100, 100)
                    #     cv2.moveWindow('bgr_axis', 100, 200)
                    #     cv2.moveWindow('depth_bgr', 500, 100)
                    #     cv2.moveWindow('hanging_points_depth_bgr', 500, 200)
                    #     cv2.moveWindow('annotation', 900, 100)

                    cv2.imwrite(
                        os.path.join(
                            save_dir, 'color', '{:05}.png'.format(
                                data_id)), bgr)
                    cv2.imwrite(
                        os.path.join(
                            save_dir, 'color_raw', '{:05}.png'.format(
                                data_id)), bgr_raw)
                    cv2.imwrite(
                        os.path.join(
                            save_dir, 'debug_axis', '{:05}.png'.format(
                                data_id)), bgr_axis)
                    cv2.imwrite(
                        os.path.join(
                            save_dir, 'debug_heatmap', '{:05}.png'.format(
                                data_id)), bgr_annotation)
                    np.save(
                        os.path.join(
                            save_dir, 'depth', '{:05}'.format(data_id)), depth)
                    np.save(
                        os.path.join(
                            save_dir,
                            'hanging_points_depth', '{:05}'.format(data_id)),
                        hanging_points_depth)
                    np.save(
                        os.path.join(
                            save_dir, 'rotations', '{:05}'.format(data_id)),
                        rotations)
                    cv2.imwrite(
                        os.path.join(
                            save_dir, 'depth_bgr', '{:05}.png'.format(
                                data_id)), depth_bgr)
                    cv2.imwrite(
                        os.path.join(
                            save_dir, 'heatmap', '{:05}.png'.format(
                                data_id)), annotation_img)
                    np.save(
                        os.path.join(
                            save_dir, 'clip_info', '{:05}'.format(data_id)),
                        clip_info)

                    data_count += 1
                    data_id += 1
                    r.remove_all_objects()
                r.remove_all_objects()
    except KeyboardInterrupt:
        pass
