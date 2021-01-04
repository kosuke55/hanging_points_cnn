#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os.path as osp
import sys
import yaml
from pathlib import Path

import cameramodels
import open3d as o3d
import numpy as np
import skrobot
import torch
import trimesh
from skrobot.coordinates.math import matrix2quaternion
from skrobot.coordinates.math import rotation_matrix_from_axis
from torchvision import transforms

from hanging_points_cnn.learning_scripts.hpnet import HPNET
from hanging_points_cnn.utils.image import draw_roi
from hanging_points_cnn.utils.image import normalize_depth
from hanging_points_cnn.utils.image import unnormalize_depth
from hanging_points_cnn.utils.image import overlay_heatmap
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


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input-dir', '-i', type=str,
        help='input directory',
        default=None)
    parser.add_argument(
        '--color', '-c', type=str,
        help='color image (.png)', default=None)
    parser.add_argument(
        '--depth', '-d', type=str,
        help='depth image (.npy)', default=None)
    parser.add_argument(
        '--camera-info', '-ci', type=str,
        help='camera info file (.yaml)', default=None)

    parser.add_argument(
        '--pretrained_model',
        '-p',
        type=str,
        help='Pretrained models',
        # default='/media/kosuke55/SANDISK-2/meshdata/shapenet_hanging_render/1014/hpnet_latestmodel_20201018_0109.pt') # shapenet
        # default='/media/kosuke55/SANDISK-2/meshdata/random_shape_shapenet_hanging_render/1010/hpnet_latestmodel_20201016_0453.pt')  # gan
        # default='/media/kosuke55/SANDISK-2/meshdata/shapenet_pouring_render/1218_mug_cap_helmet_bowl/hpnet_latestmodel_20201218_1032.pt')  # noqa
        default='/media/kosuke55/SANDISK-2/meshdata/shapenet_pouring_render/1218_mug_cap_helmet_bowl/hpnet_latestmodel_20201219_0213.pt')  # noqa
    parser.add_argument(
        '--predict-depth', '-pd', type=int,
        help='predict-depth', default=0)
    parser.add_argument(
        '--task', '-t', type=str,
        help='h(hanging) or p(pouring)',
        default='h')

    args = parser.parse_args()
    base_dir = args.input_dir
    pretrained_model = args.pretrained_model

    config_path = str(Path(osp.abspath(__file__)).parent.parent
                      / 'learning_scripts' / 'config' / 'gray_model.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    task_type = args.task
    if task_type == 'p':
        task_type = 'pouring'
    else:
        task_type = 'hanging'

    target_size = tuple(config['target_size'])
    depth_range = config['depth_range']
    depth_roi_size = config['depth_roi_size'][task_type]

    print('task type: {}'.format(task_type))
    print('depth roi size: {}'.format(depth_roi_size))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HPNET(config).to(device)
    model.load_state_dict(torch.load(pretrained_model), strict=False)
    model.eval()

    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))

    if base_dir is not None:
        color_paths = list(Path(base_dir).glob('**/color/*.png'))
    elif args.color is not None \
            and args.depth is not None \
            and args.camera_info is not None:
        color_paths = [args.color]

    else:
        return False

    is_first_loop = True
    try:
        for color_path in color_paths:
            if not is_first_loop:
                viewer.delete(pc)  # noqa
                for c in contact_point_sphere_list:  # noqa
                    viewer.delete(c)

            if base_dir is not None:
                camera_info_path = str(
                    (color_path.parent.parent /
                     'camera_info' /
                     color_path.stem).with_suffix('.yaml'))
                depth_path = str(
                    (color_path.parent.parent /
                     'depth' /
                     color_path.stem).with_suffix('.npy'))
                color_path = str(color_path)
            else:
                camera_info_path = args.camera_info
                color_path = args.color
                depth_path = args.depth

            camera_model = cameramodels.PinholeCameraModel.from_yaml_file(
                camera_info_path)
            camera_model.target_size = target_size

            cv_bgr = cv2.imread(color_path)

            intrinsics = camera_model.open3d_intrinsic

            cv_bgr = cv2.resize(cv_bgr, target_size)
            cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
            color = o3d.geometry.Image(cv_rgb)

            cv_depth = np.load(depth_path)
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

            viewer.add(pc)

            if config['use_bgr2gray']:
                gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, target_size)[..., None] / 255.
                normalized_depth = normalize_depth(
                    cv_depth, depth_range[0], depth_range[1])[..., None]
                in_feature = np.concatenate(
                    (normalized_depth, gray), axis=2).astype(np.float32)
            else:
                raise NotImplementedError()

            transform = transforms.Compose([
                transforms.ToTensor()])
            in_feature = transform(in_feature)

            in_feature = in_feature.to(device)
            in_feature = in_feature.unsqueeze(0)

            confidence, depth, rotation = model(in_feature)

            confidence = confidence[0, 0:1, ...]
            confidence_np = confidence.cpu().detach().numpy().copy() * 255
            confidence_np = confidence_np.transpose(1, 2, 0)
            confidence_np[confidence_np <= 0] = 0
            confidence_np[confidence_np >= 255] = 255
            confidence_img = confidence_np.astype(np.uint8)

            print(model.rois_list)
            contact_point_sphere_list = []
            roi_image = cv_bgr.copy()
            for i, (roi, roi_center) in enumerate(
                    zip(model.rois_list[0], model.rois_center_list[0])):
                if roi.tolist() == [0, 0, 0, 0]:
                    continue
                roi = roi.cpu().detach().numpy().copy()
                roi_image = draw_roi(roi_image, roi)
                hanging_point_x = roi_center[0]
                hanging_point_y = roi_center[1]
                v = rotation[i].cpu().detach().numpy()
                v /= np.linalg.norm(v)
                rot = rotation_matrix_from_axis(v, [0, 1, 0], 'xy')
                q = matrix2quaternion(rot)

                hanging_point = np.array(
                    camera_model.project_pixel_to_3d_ray(
                        [int(hanging_point_x), int(hanging_point_y)]))

                if args.predict_depth:
                    dep = depth[i].cpu().detach().numpy().copy()
                    dep = unnormalize_depth(
                        dep, depth_range[0], depth_range[1]) * 0.001
                    length = float(dep) / hanging_point[2]
                else:
                    depth_roi = make_box(
                        roi_center,
                        width=depth_roi_size[1],
                        height=depth_roi_size[0],
                        img_shape=target_size,
                        xywh=False)
                    depth_roi_clip = cv_depth[
                        depth_roi[0]:depth_roi[2],
                        depth_roi[1]:depth_roi[3]]

                    dep_roi_clip = depth_roi_clip[np.where(
                        np.logical_and(
                            config['depth_range'][0] < depth_roi_clip,
                            depth_roi_clip < config['depth_range'][1]))]

                    dep_roi_clip = np.median(dep_roi_clip) * 0.001

                    if dep_roi_clip == np.nan:
                        continue
                    length = float(dep_roi_clip) / hanging_point[2]

                hanging_point *= length

                contact_point_sphere = skrobot.model.Sphere(
                    0.001, color=[255, 0, 0])
                contact_point_sphere.newcoords(
                    skrobot.coordinates.Coordinates(pos=hanging_point, rot=q))
                viewer.add(contact_point_sphere)
                contact_point_sphere_list.append(contact_point_sphere)

            if is_first_loop:
                viewer.show()

            heatmap = overlay_heatmap(cv_bgr, confidence_img)
            cv2.imshow('heatmap', heatmap)
            cv2.imshow('roi', roi_image)
            print('Next data: [ENTER] on image window.\n'
                  'Quit: [q] on image window.')
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord('q'):
                break

            is_first_loop = False

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
