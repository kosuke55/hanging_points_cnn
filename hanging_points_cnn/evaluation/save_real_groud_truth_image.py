"""Visualize the manual annotation and save images

saved images are pointcloud, colored depth, and heatmap/
"""

from __future__ import division

import argparse
import os
import os.path as osp
import re
from pathlib import Path

import cameramodels
import cv2
import numpy as np
import skrobot
import open3d as o3d
import trimesh

from hanging_points_generator.generator_utils import load_json
from hanging_points_cnn.utils.image import create_gradient_circle
from hanging_points_cnn.utils.image import colorize_depth
from hanging_points_cnn.utils.image import overlay_heatmap


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--input-dir', '-i', type=str,
    help='input directory'
    'input dir which has each category and '
    'their color, depth, camera_info data.',
    # default='/home/kosuke55/catkin_ws/src/hanging_points_cnn/data/ycb_real_eval')  # hanging real # noqa
    default='/home/kosuke55/catkin_ws/src/hanging_points_cnn/data/ycb_real_eval_pouring')  # pouring real # noqa
parser.add_argument(
    '--annotation-dir', '-a', type=str,
    help='annotation directory.',
    # default='/media/kosuke55/SANDISK/meshdata/ycb_hanging_object/real_ycb_annotation_1027')  # hanging real # noqa
    default='/media/kosuke55/SANDISK/meshdata/ycb_pouring_object_16/real_ycb_annotation_pouring')  # pouring real # noqa
parser.add_argument(
    '--save-dir', '-s', type=str,
    help='directory to save PointCloud with annotation, colorized depth and heatmap. '
    '<input_dir>/<save_dir>',
    # default='groud_truth_hanging')
    default='groud_truth_pouring')
parser.add_argument(
    '--visualize-image', '-vi', action='store_true',
    help='visualize image')
parser.add_argument(
    '--reverse', '-r', action='store_true',
    help='reverse x direction')

args = parser.parse_args()

input_dir = args.input_dir
annotation_dir = args.annotation_dir
save_dir = str(Path(input_dir) / args.save_dir)
visualize_image = args.visualize_image
reverse_x = args.reverse

os.makedirs(save_dir, exist_ok=True)
print(save_dir)

regex = re.compile(r'\d+')

target_size = (256, 256)
color_paths = list(sorted(Path(input_dir).glob('*/color/*.png')))
try:
    for color_path in color_paths:
        camera_info_path = color_path.parent.parent / \
            'camera_info' / color_path.with_suffix('.yaml').name
        depth_path = color_path.parent.parent / \
            'depth' / color_path.with_suffix('.npy').name

        camera_model = cameramodels.PinholeCameraModel.from_yaml_file(
            str(camera_info_path))
        camera_model.target_size = target_size
        intrinsics = camera_model.open3d_intrinsic

        cv_bgr = cv2.imread(str(color_path))
        cv_bgr = cv2.resize(cv_bgr, target_size)
        cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        color = o3d.geometry.Image(cv_rgb)

        cv_depth = np.load(str(depth_path))
        cv_depth = cv2.resize(cv_depth, target_size,
                              interpolation=cv2.INTER_NEAREST)
        depth = o3d.geometry.Image(cv_depth)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsics)
        trimesh_pc = trimesh.PointCloud(
            np.asarray(
                pcd.points), np.asarray(
                pcd.colors))
        pc = skrobot.model.PointCloudLink(trimesh_pc)

        idx = int(regex.findall(color_path.stem)[0])

        category = color_path.parent.parent.name
        pose_file = osp.join(annotation_dir, category + '_{}.json'.format(idx))
        contact_points_dict = load_json(str(pose_file))
        contact_points = contact_points_dict['contact_points']
        contact_point_marker_list = []
        for i, cp in enumerate(contact_points):
            contact_point_marker = skrobot.model.Axis(0.003, 0.05)
            contact_point_marker.newcoords(
                skrobot.coordinates.Coordinates(pos=cp[0], rot=cp[1:]))
            if reverse_x:
                contact_point_marker.rotate(np.pi, 'y')

            contact_point_marker_list.append(contact_point_marker)

        confidence = np.zeros(cv_bgr.shape[:2], dtype=np.uint32)
        for cp in contact_point_marker_list:
            px, py = camera_model.project3d_to_pixel(cp.worldpos())
            create_gradient_circle(
                confidence, py, px)
        confidence = (confidence / confidence.max() * 255).astype(np.uint8)

        heatmap = overlay_heatmap(cv_bgr, confidence.astype(np.uint8))
        depth_bgr = colorize_depth(cv_depth, ignore_value=0)

        viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 640))
        viewer.add(pc)
        for contact_point_marker in contact_point_marker_list:
            viewer.add(contact_point_marker)
        viewer._init_and_start_app()

        skip = False
        if visualize_image:
            cv2.imshow('heatmp', heatmap)
            cv2.imshow('depth', depth_bgr)
            cv2.waitKey()

            if cv2.waitKey(0) != ord('y'):
                print('skip save image')
                skip = True
            cv2.destroyAllWindows()

        if not skip:
            cv2.imwrite(
                osp.join(
                    save_dir,
                    category +
                    '_heatmap_{}.png'.format(idx)),
                heatmap)
            cv2.imwrite(
                osp.join(
                    save_dir,
                    category +
                    '_depth_bgr_{}.png'.format(idx)),
                depth_bgr)

        from PIL import Image
        loop = True
        while loop and not skip:
            try:
                data = viewer.scene.save_image(visible=True)
                rendered = Image.open(trimesh.util.wrap_as_stream(data))
                rendered.save(
                    osp.join(
                        save_dir,
                        category +
                        '_hp_{}.png'.format(idx)))
                loop = False
            except AttributeError:
                print('Fail to save pcd image. Try again.')

except KeyboardInterrupt:
    pass
