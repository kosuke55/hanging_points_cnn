import argparse
import os
import sys
from pathlib import Path

import cameramodels
import open3d as o3d
import numpy as np
import skrobot
import torch
import trimesh
from hanging_points_generator.generator_utils import load_json
from skrobot.coordinates.math import matrix2quaternion
from skrobot.coordinates.math import rotation_matrix_from_axis
from skrobot.coordinates.math import quaternion2matrix
from hanging_points_generator.generator_utils import save_json
from torchvision import transforms

from hanging_points_cnn.learning_scripts.hpnet import HPNET
from hanging_points_cnn.utils.image import draw_axis
from hanging_points_cnn.utils.image import draw_roi
from hanging_points_cnn.utils.image import normalize_depth
from hanging_points_cnn.utils.image import unnormalize_depth
from hanging_points_cnn.utils.image import overlay_heatmap


try:
    import cv2
except ImportError:
    for path in sys.path:
        if '/opt/ros/' in path:
            print('sys.path.remove({})'.format(path))
            sys.path.remove(path)
            import cv2
            sys.path.append(path)


def quaternion2xvec(q):
    m = quaternion2matrix(q)
    return m[:, 0]


def two_vectors_angle(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(cos)


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--input-dir', '-i', type=str,
    help='input urdf',
    default='/media/kosuke55/SANDISK-2/meshdata/ycb_eval')

parser.add_argument(
    '--pretrained_model',
    '-p',
    type=str,
    help='Pretrained models',
    # default='/media/kosuke55/SANDISK-2/meshdata/shapenet_hanging_render/1014/hpnet_latestmodel_20201018_0109.pt')  # shapenet
    default='/media/kosuke55/SANDISK-2/meshdata/random_shape_shapenet_hanging_render/1010/hpnet_latestmodel_20201016_0453.pt')  # gan
parser.add_argument(
    '--predict-depth', '-pd', type=int,
    help='predict-depth', default=0)

args = parser.parse_args()
base_dir = args.input_dir
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
thresh_distance = 0.03

color_paths = Path(base_dir).glob('*/*/color/*.png')
first = True
try:
    for color_path in color_paths:
        if not first:
            viewer.delete(pc)
            for c in contact_point_sphere_list:
                viewer.delete(c)
        annotation_path = color_path.parent.parent / \
            'annotation' / color_path.with_suffix('.json').name
        camera_info_path = color_path.parent.parent / \
            'camera_info' / color_path.with_suffix('.yaml').name
        depth_path = color_path.parent.parent / \
            'depth' / color_path.with_suffix('.npy').name
        # color_path = str(color_path)

        annotation_data = load_json(str(annotation_path))
        camera_model = cameramodels.PinholeCameraModel.from_yaml_file(
            str(camera_info_path))
        camera_model.target_size = (256, 256)
        intrinsics = camera_model.open3d_intrinsic

        cv_bgr = cv2.imread(str(color_path))
        cv_bgr = cv2.resize(cv_bgr, (256, 256))
        cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        color = o3d.geometry.Image(cv_rgb)

        cv_depth = np.load(str(depth_path))
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

        contact_point_sphere_list = []
        pos_list = []
        quaternion_list = []
        roi_image = cv_bgr.copy()
        axis_image = cv_bgr.copy()
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

            camera_model_crop_resize \
                = camera_model.crop_resize_camera_info(target_size=[256, 256])

            hanging_point = np.array(
                camera_model_crop_resize.project_pixel_to_3d_ray(
                    [int(hanging_point_x), int(hanging_point_y)]))

            draw_axis(axis_image,
                      rot,
                      hanging_point,
                      camera_model.K)
            if args.predict_depth:
                dep = depth[i].cpu().detach().numpy().copy()
                dep = unnormalize_depth(
                    dep, depth_range[0], depth_range[1]) * 0.001
                length = float(dep) / hanging_point[2]
            else:
                depth_roi_clip = cv_depth[
                    roi_center[1] - 10:roi_center[1] + 10,
                    roi_center[0] - 10:roi_center[0] + 10]
                dep_roi_clip = depth_roi_clip[np.where(
                    np.logical_and(config['depth_range'][0] < depth_roi_clip,
                                   depth_roi_clip < config['depth_range'][1]))]
                dep_roi_clip = np.median(dep_roi_clip) * 0.001
                if dep_roi_clip == np.nan:
                    continue
                length = float(dep_roi_clip) / hanging_point[2]

            hanging_point *= length
            pos_list.append(hanging_point)
            quaternion_list.append(q)

            contact_point_sphere = skrobot.models.Sphere(
                0.001, color=[255, 0, 0])
            contact_point_sphere.newcoords(
                skrobot.coordinates.Coordinates(pos=hanging_point, rot=q))
            viewer.add(contact_point_sphere)
            contact_point_sphere_list.append(contact_point_sphere)

        heatmap = overlay_heatmap(cv_bgr, confidence_img)

        gt_pos_list = []
        gt_quaternon_list = []
        gt_labels = []
        for annotation in annotation_data:
            cx = annotation['xy'][0]
            cy = annotation['xy'][1]
            q = np.array(annotation['quaternion'])
            dep = annotation['depth']
            label = annotation['label']
            pos = np.array(
                camera_model.project_pixel_to_3d_ray([cx, cy]))
            length = dep * 0.001 / pos[2]
            pos = pos * length
            gt_pos_list.append(pos)
            gt_quaternon_list.append(q)
            gt_labels.append(label)

        # Calculate diff
        diff_dict = {}
        diff_dict['-1'] = {}
        diff_dict['-1']['pos_diff'] = []
        diff_dict['-1']['distance'] = []
        diff_dict['-1']['angle'] = []

        for pos, quaternion in zip(pos_list, quaternion_list):
            pos = np.array(pos)
            pos_diff = pos - gt_pos_list
            distances = np.linalg.norm(pos_diff, axis=1)
            min_idx = np.argmin(distances)
            min_distance = np.min(distances)

            if str(gt_labels[min_idx]) not in diff_dict:
                diff_dict[str(gt_labels[min_idx])] = {}
                diff_dict[str(gt_labels[min_idx])]['pos_diff'] = []
                diff_dict[str(gt_labels[min_idx])]['distance'] = []
                diff_dict[str(gt_labels[min_idx])]['angle'] = []

            pos_diff = pos_diff[min_idx].tolist()
            vec = quaternion2xvec(quaternion)
            gt_vec = quaternion2xvec(gt_quaternon_list[min_idx])
            angle = min(two_vectors_angle(vec, gt_vec),
                        two_vectors_angle(vec, -gt_vec))

            if min_distance > thresh_distance:
                diff_dict['-1']['pos_diff'].append(pos_diff)
                diff_dict['-1']['distance'].append(min_distance)
                diff_dict['-1']['angle'].append(angle)
            else:
                diff_dict[str(gt_labels[min_idx])]['pos_diff'].append(pos_diff)
                diff_dict[str(gt_labels[min_idx])
                          ]['distance'].append(min_distance)
                diff_dict[str(gt_labels[min_idx])]['angle'].append(angle)
        
        print('\n\n')
        print(color_path)
        for key in diff_dict:
            print('----label %s ----' % key)
            print('len: %d' % len(diff_dict[key]['angle']))
            if key == '-1':
                continue
            pos_diff = np.array(diff_dict[key]['pos_diff'])
            if pos_diff.size == 0:
                continue
            diff_dict[key]['pos_diff_mean'] = np.mean(
                pos_diff, axis=0).tolist()
            diff_dict[key]['pos_diff_max'] = np.max(pos_diff, axis=0).tolist()
            diff_dict[key]['pos_diff_min'] = np.min(pos_diff, axis=0).tolist()

            print('pos_diff_max %f %f %f' %
                  tuple(diff_dict[key]['pos_diff_max']))
            print('pos_diff_mean %f %f %f' %
                  tuple(diff_dict[key]['pos_diff_mean']))
            print('pos_diff_min %f %f %f' %
                  tuple(diff_dict[key]['pos_diff_min']))

            angle = np.array(diff_dict[key]['angle'])
            diff_dict[key]['angle_mean'] = np.mean(angle).tolist()
            diff_dict[key]['angle_max'] = np.max(angle).tolist()
            diff_dict[key]['angle_min'] = np.min(angle).tolist()

            print('angle_max %f' % diff_dict[key]['angle_max'])
            print('angle_mean %f' % diff_dict[key]['angle_mean'])
            print('angle_min %f' % diff_dict[key]['angle_min'])

        # pred_dir = color_path.parent.parent.parent.parent / 'pred_shapenet'
        pred_dir = color_path.parent.parent.parent.parent / 'pred'
        pred_heatmap_dir = pred_dir / 'heatmap'
        pred_diff_dir = pred_dir / 'diff'
        pred_axis_dir = pred_dir / 'axis'
        category_name = color_path.parent.parent.parent.name
        os.makedirs(str(pred_dir), exist_ok=True)
        os.makedirs(str(pred_heatmap_dir), exist_ok=True)
        os.makedirs(str(pred_axis_dir), exist_ok=True)
        os.makedirs(str(pred_diff_dir), exist_ok=True)
        cv2.imwrite(str(pred_heatmap_dir / Path(
            category_name +
            '_' +
            color_path.name)), heatmap)
        cv2.imwrite(str(pred_axis_dir / Path(
            category_name +
            '_' +
            color_path.name)), axis_image)
        save_json(str(pred_diff_dir / Path(
            category_name +
            '_' +
            color_path.with_suffix('.json').name)), diff_dict)

        if first:
            viewer.show()
            first = False
        cv2.imshow('heatmap', heatmap)
        cv2.imshow('roi', roi_image)
        cv2.imshow('axis', axis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


except KeyboardInterrupt:
    pass
