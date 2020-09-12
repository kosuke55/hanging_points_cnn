import argparse
import os.path as osp
import sys

import cameramodels
import open3d as o3d
import numpy as np
import skrobot
import trimesh

from hanging_points_cnn.utils.rois_tools import find_rois

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
    default='/media/kosuke55/SANDISK-2/meshdata/ycb_hanging_object/0808/pocky-2020-08-08-18-44-42-555262-14432')

parser.add_argument(
    '--idx', type=int,
    help='data idx',
    default=0)

args = parser.parse_args()
base_dir = args.input_dir
idx = args.idx

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))

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
# o3d.visualization.draw_geometries([pcd])

rotations_path = osp.join(
    base_dir, 'rotations', '{:06}.npy'.format(idx))
rotations = np.load(rotations_path)

hanging_points_depth_path = osp.join(
    base_dir, 'hanging_points_depth', '{:06}.npy'.format(idx))
hanging_points_depth = np.load(hanging_points_depth_path)

confidence_path = osp.join(
    base_dir, 'heatmap', '{:06}.png'.format(idx))
confidence = cv2.imread(
    confidence_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.

_, rois_center = find_rois(confidence)
rois_center = rois_center[0]
for roi_center in rois_center:
    cx, cy = roi_center
    q = rotations[cy, cx]
    dep = hanging_points_depth[cy, cx]
    print(cx, cy)
    pos = np.array(
        cameramodel.project_pixel_to_3d_ray([cx, cy])) * dep * 0.001
    contact_point_sphere = skrobot.models.Sphere(0.001, color=[255, 0, 0])
    contact_point_sphere.newcoords(
        skrobot.coordinates.Coordinates(pos=pos, rot=q))
    viewer.add(contact_point_sphere)

trimesh_pc = trimesh.PointCloud(np.asarray(pcd.points), np.asarray(pcd.colors))
viewer.scene.add_geometry(trimesh_pc)

viewer.show()
