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
    default='/media/kosuke55/SANDISK-2/meshdata/ycb_hanging_object/0808/pocky-2020-08-08-18-44-42-555262-14432')

parser.add_argument(
    '--idx', type=int,
    help='data idx',
    default=0)

parser.add_argument(
    '--pretrained_model',
    '-p',
    type=str,
    help='Pretrained model',
    default='/media/kosuke/SANDISK/hanging_points_net/checkpoints/gray/hpnet_bestmodel_20200827_0552.pt')

args = parser.parse_args()
base_dir = args.input_dir
idx = args.idx
pretrained_model = args.pretrained_model

camera_info_path = osp.join(
    base_dir, 'camera_info', '{:06}.yaml'.format(idx))
camera_model = cameramodels.PinholeCameraModel.from_yaml_file(
    camera_info_path)
intrinsics = camera_model.open3d_intrinsic

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))

color_path = osp.join(base_dir, 'color', '{:06}.png'.format(idx))
cv_bgr = cv2.imread(color_path)
cv_bgr = cv2.resize(cv_bgr, (256, 256))
cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
color = o3d.geometry.Image(cv_rgb)
# color = o3d.io.read_image(color_path)

depth_path = osp.join(base_dir, 'depth', '{:06}.npy'.format(idx))
cv_depth = np.load(depth_path)
cv_depth = cv2.resize(cv_depth, (256, 256),
                      interpolation=cv2.INTER_NEAREST)

depth = o3d.geometry.Image(cv_depth)
# depth = o3d.geometry.Image(np.load(depth_path))

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd, intrinsics)
# o3d.visualization.draw_geometries([pcd])

config = {
    'output_channels': 1,
    'feature_extractor_name': 'resnet50',
    'confidence_thresh': 0.3,
    'depth_range': [200, 1500],
    'use_bgr': True,
    'use_bgr2gray': True,
    'roi_padding': 50
}
depth_range = config['depth_range']

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HPNET(config).to(device)
model.load_state_dict(torch.load(pretrained_model))
model.eval()

in_feature = in_feature.to(device)
in_feature = in_feature.unsqueeze(0)

confidence, depth_and_rotation = model(in_feature)

confidence = confidence[0, 0:1, ...]
confidence_np = confidence.cpu().detach().numpy().copy() * 255
confidence_np = confidence_np.transpose(1, 2, 0)
confidence_np[confidence_np <= 0] = 0
confidence_np[confidence_np >= 255] = 255
confidence_img = confidence_np.astype(np.uint8)

print(model.rois_list)
for i, roi in enumerate(model.rois_list[0]):
    if roi.tolist() == [0, 0, 0, 0]:
        continue
    roi = roi.cpu().detach().numpy().copy()
    cv_bgr = draw_roi(cv_bgr, roi)
    hanging_point_x = int((roi[0] + roi[2]) / 2)
    hanging_point_y = int((roi[1] + roi[3]) / 2)

    v = depth_and_rotation[i, 1:4].cpu().detach().numpy()
    v /= np.linalg.norm(v)
    rot = rotation_matrix_from_axis(v, [0, 1, 0], 'xy')
    q = matrix2quaternion(rot)

    camera_model_crop_resize \
        = camera_model.crop_resize_camera_info(target_size=[256, 256])

    hanging_point = np.array(
        camera_model_crop_resize.project_pixel_to_3d_ray(
            [int(hanging_point_x),
             int(hanging_point_y)]))

    dep = depth_and_rotation[i, 0].cpu().detach().numpy().copy()
    dep = unnormalize_depth(
        dep, depth_range[0], depth_range[1]) * 0.001
    length = float(dep) / hanging_point[2]

    hanging_point *= length

    contact_point_sphere = skrobot.models.Sphere(0.001, color=[255, 0, 0])
    contact_point_sphere.newcoords(
        skrobot.coordinates.Coordinates(pos=hanging_point, rot=q))
    viewer.add(contact_point_sphere)

trimesh_pc = trimesh.PointCloud(np.asarray(pcd.points), np.asarray(pcd.colors))
viewer.scene.add_geometry(trimesh_pc)

viewer.show()

cv2.imshow('confidence', confidence_img)
cv2.imshow('roi', cv_bgr)
cv2.waitKey()
