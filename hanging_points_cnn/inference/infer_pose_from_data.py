import argparse
import os.path as osp
import sys

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
    default=None)

parser.add_argument(
    '--idx', type=int,
    help='data idx',
    default=0)

parser.add_argument(
    '--color', '-c', type=str,
    help='color', default=None)
parser.add_argument(
    '--depth', '-d', type=str,
    help='depth', default=None)
parser.add_argument(
    '--camera-info', '-ci', type=str,
    help='camera info', default=None)

parser.add_argument(
    '--pretrained_model',
    '-p',
    type=str,
    help='Pretrained modesel',
    default='/media/kosuke/SANDISK/hanging_points_net/checkpoints/gray/hpnet_latestmodel_20200923_1755.pt')
parser.add_argument(
    '--predict-depth', '-pd', type=int,
    help='predict-depth', default=1)

args = parser.parse_args()
base_dir = args.input_dir
start_idx = args.idx
pretrained_model = args.pretrained_model

config = {
    'output_channels': 1,
    'feature_extractor_name': 'resnet50',
    'confidence_thresh': 0.3,
    'depth_range': [100, 1500],
    'use_bgr': True,
    'use_bgr2gray': True,
    'roi_padding': 50
}
depth_range = config['depth_range']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HPNET(config).to(device)
model.load_state_dict(torch.load(pretrained_model))
model.eval()

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))

try:
    for idx in range(start_idx, 100000):
        print(idx)
        if idx != start_idx:
            viewer.delete(pc)
            for c in contact_point_sphere_list:
                viewer.delete(c)

        if base_dir is not None:
            camera_info_path = osp.join(
                base_dir, 'camera_info', '{:06}.yaml'.format(idx))
            color_path = osp.join(base_dir, 'color', '{:06}.png'.format(idx))
            depth_path = osp.join(base_dir, 'depth', '{:06}.npy'.format(idx))
        else:
            camera_info_path = args.camera_info
            color_path = args.color
            depth_path = args.depth

        camera_model = cameramodels.PinholeCameraModel.from_yaml_file(
            camera_info_path)
        camera_model.target_size = (256, 256)
        intrinsics = camera_model.open3d_intrinsic

        cv_bgr = cv2.imread(color_path)
        # cv_bgr = cv2.flip(cv_bgr, 0)
        cv_bgr = cv2.resize(cv_bgr, (256, 256))
        cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        color = o3d.geometry.Image(cv_rgb)
        # color = o3d.io.read_image(color_path)

        cv_depth = np.load(depth_path)
        # cv_depth = cv2.flip(cv_depth, 0)
        # cv_depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        cv_depth = cv2.resize(cv_depth, (256, 256),
                              interpolation=cv2.INTER_NEAREST)
        depth = o3d.geometry.Image(cv_depth)
        # depth = o3d.geometry.Image(np.load(depth_path))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsics)
        trimesh_pc = trimesh.PointCloud(
            np.asarray(
                pcd.points), np.asarray(
                pcd.colors))
        pc = skrobot.models.PointCloudLink(trimesh_pc)

        viewer.add(pc)
        # o3d.visualization.draw_geometries([pcd])

        if config['use_bgr2gray']:
            gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (256, 256))[..., None] / 255.
            normalized_depth = normalize_depth(
                cv_depth, depth_range[0], depth_range[1])[..., None]
            in_feature = np.concatenate(
                (normalized_depth, gray), axis=2).astype(np.float32)
        else:
            raise

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
        for i, roi in enumerate(model.rois_list[0]):
            if roi.tolist() == [0, 0, 0, 0]:
                continue
            roi = roi.cpu().detach().numpy().copy()
            cv_bgr = draw_roi(cv_bgr, roi)
            hanging_point_x = int((roi[0] + roi[2]) / 2)
            hanging_point_y = int((roi[1] + roi[3]) / 2)
            v = rotation[i].cpu().detach().numpy()
            v /= np.linalg.norm(v)
            rot = rotation_matrix_from_axis(v, [0, 1, 0], 'xy')
            q = matrix2quaternion(rot)

            camera_model_crop_resize \
                = camera_model.crop_resize_camera_info(target_size=[256, 256])

            hanging_point = np.array(
                camera_model_crop_resize.project_pixel_to_3d_ray(
                    [int(hanging_point_x), int(hanging_point_y)]))

            if args.predict_depth:
                dep = depth[i].cpu().detach().numpy().copy()
                dep = unnormalize_depth(
                    dep, depth_range[0], depth_range[1]) * 0.001
                length = float(dep) / hanging_point[2]
            else:
                depth_roi_clip = cv_depth[int(roi[1]):int(roi[3]),
                                          int(roi[0]):int(roi[2])]
                dep_roi_clip = depth_roi_clip[np.where(
                    np.logical_and(config['depth_range'][0] < depth_roi_clip,
                                   depth_roi_clip < config['depth_range'][1]))]
                dep_roi_clip = np.median(dep_roi_clip) * 0.001
                if dep_roi_clip == np.nan:
                    continue
                length = float(dep_roi_clip) / hanging_point[2]

            hanging_point *= length

            contact_point_sphere = skrobot.models.Sphere(
                0.001, color=[255, 0, 0])
            contact_point_sphere.newcoords(
                skrobot.coordinates.Coordinates(pos=hanging_point, rot=q))
            viewer.add(contact_point_sphere)
            contact_point_sphere_list.append(contact_point_sphere)

        if idx == start_idx:
            viewer.show()

        cv2.imshow('confidence', confidence_img)
        cv2.imshow('roi', cv_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
except KeyboardInterrupt:
    pass
