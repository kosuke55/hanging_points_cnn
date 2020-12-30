import argparse
import os
from pathlib import Path

import cameramodels
import cv2
import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--input', '-i', type=str,
    help='input directory '
    '<input_directory>/<object>/<color, depth, camra_info>/<png, npy, yaml>',
    default='/home/kosuke55/catkin_ws/src/hanging_points_cnn/data/ycb_real_eval_pouring')  # noqa
parser.add_argument(
    '--output', '-o', type=str,
    help='output directory '
    '<output_directory>/<pcd>',
    default='/media/kosuke55/SANDISK/meshdata/ycb_pouring_object_16/real_ycb_annotation_pouring')  # noqa
args = parser.parse_args()

color_paths = list(sorted(Path(args.input).glob('*/color/*.png')))
out_dir = args.output
os.makedirs(str(out_dir), exist_ok=True)

target_size = (256, 256)
for color_path in color_paths:
    cv_bgr = cv2.imread(str(color_path))
    cv_bgr = cv2.resize(cv_bgr, target_size)
    cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
    color = o3d.geometry.Image(cv_rgb)

    depth_path = color_path.parent.parent / \
        'depth' / color_path.with_suffix('.npy').name
    cv_depth = np.load(str(depth_path))
    cv_depth = cv2.resize(cv_depth, target_size,
                          interpolation=cv2.INTER_NEAREST)
    depth = o3d.geometry.Image(cv_depth)

    camera_info_path = color_path.parent.parent / \
        'camera_info' / color_path.with_suffix('.yaml').name
    camera_model = cameramodels.PinholeCameraModel.from_yaml_file(
        str(camera_info_path))
    camera_model.target_size = target_size
    intrinsics = camera_model.open3d_intrinsic

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsics)
    # o3d.visualization.draw_geometries([pcd])

    idx = int(color_path.stem)
    category = color_path.parent.parent.name
    pcd_name = str(Path(out_dir) / category) + '_{}.pcd'.format(idx)
    print(pcd_name)
    o3d.io.write_point_cloud(pcd_name, pcd)
