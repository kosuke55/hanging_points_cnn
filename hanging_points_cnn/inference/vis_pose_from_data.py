import argparse
import os.path as osp
from six.moves import input
import sys
import time

import cameramodels
import open3d as o3d
import numpy as np
import skrobot
import trimesh
from hanging_points_generator.generator_utils import load_json

from hanging_points_cnn.utils.rois_tools import find_rois


def label_colormap(n_label=256, value=None):
    """Label colormap.

    original code is https://github.com/wkentaro/imgviz/blob/master/imgviz/label.py

    Parameters
    ----------
    n_labels: int
        Number of labels (default: 256).
    value: float or int
        Value scale or value of label color in HSV space.

    Returns
    -------
    cmap: numpy.ndarray, (N, 3), numpy.uint8
        Label id to colormap.

    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n_label, 3), dtype=np.uint8)
    for i in range(0, n_label):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b

    if value is not None:
        hsv = color_module.rgb2hsv(cmap.reshape(1, -1, 3))
        if isinstance(value, float):
            hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
        else:
            assert isinstance(value, int)
            hsv[:, 1:, 2] = value
        cmap = color_module.hsv2rgb(hsv).reshape(-1, 3)
    return cmap


try:
    import cv2
except ImportError:
    for path in sys.path:
        if '/opt/ros/' in path:
            print('sys.path.remove({})'.format(path))
            sys.path.remove(path)
            import cv2
            sys.path.append(path)

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--input-dir', '-i', type=str,
    help='input urdf',
    default='/media/kosuke55/SANDISK-2/meshdata/ycb_eval/019_pitcher_base/pocky-2020-10-17-06-01-16-481902-45682')

parser.add_argument(
    '--idx', type=int,
    help='data idx',
    default=0)

args = parser.parse_args()
base_dir = args.input_dir
start_idx = args.idx


viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))

for idx in range(start_idx, 100000):
    print(idx)
    if idx != start_idx:
        viewer.delete(pc)
        for c in contact_point_sphere_list:
            viewer.delete(c)

    annotation_path = osp.join(
        base_dir, 'annotation', '{:06}.json'.format(idx))
    annotation_data = load_json(annotation_path)

    color_path = osp.join(base_dir, 'color', '{:06}.png'.format(idx))
    color = o3d.io.read_image(color_path)

    depth_path = osp.join(base_dir, 'depth', '{:06}.npy'.format(idx))
    depth = np.load(depth_path)
    depth = o3d.geometry.Image(depth)

    camera_info_path = osp.join(
        base_dir, 'camera_info', '{:06}.yaml'.format(idx))
    cameramodel = cameramodels.PinholeCameraModel.from_yaml_file(
        camera_info_path)
    intrinsics = cameramodel.open3d_intrinsic
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsics)

    contact_point_sphere_list = []
    if 'label' in annotation_data[0]:
        labels = [a['label'] for a in annotation_data]
        color_map = label_colormap(max(labels) + 1)
    for annotation in annotation_data:
        cx = annotation['xy'][0]
        cy = annotation['xy'][1]
        q = np.array(annotation['quaternion'])
        dep = annotation['depth']
        color = [255, 0, 0]
        if 'label' in annotation:
            label = annotation['label']
            color = color_map[label]
        print(cx, cy)
        pos = np.array(
            cameramodel.project_pixel_to_3d_ray([cx, cy]))
        length = dep * 0.001 / pos[2]
        pos = pos * length

        contact_point_sphere = skrobot.models.Sphere(0.003, color=color)
        contact_point_sphere.newcoords(
            skrobot.coordinates.Coordinates(pos=pos, rot=q))
        viewer.add(contact_point_sphere)
        contact_point_sphere_list.append(contact_point_sphere)

    trimesh_pc = trimesh.PointCloud(
        np.asarray(
            pcd.points), np.asarray(
            pcd.colors))
    pc = skrobot.models.PointCloudLink(trimesh_pc)

    viewer.add(pc)

    if idx == start_idx:
        viewer.show()

    input('')
