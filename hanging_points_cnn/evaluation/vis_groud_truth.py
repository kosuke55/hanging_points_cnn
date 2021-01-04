from __future__ import division

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
from hanging_points_cnn.utils.image import draw_axis
from hanging_points_cnn.utils.image import create_gradient_circle

base_dir = '/home/kosuke55/catkin_ws/src/hanging_points_cnn/data/ycb_real_eval'
color_paths = list(sorted(Path(base_dir).glob('*/color/*.png')))

annotation_dir = '/media/kosuke55/SANDISK/meshdata/ycb_hanging_object/real_ycb_annotation_1027'

# save_dir = 'groud_truth_1027'
save_dir = 'hoge'
os.makedirs(save_dir, exist_ok=True)

regex = re.compile(r'\d+')
gui = True

try:
    for color_path in color_paths:
        # if 'clamp' not in str(color_path):
        #     continue
        # s.Coordinates(pos=cp[0], rot=cp[1:]).rotate(np.pi, 'y'))
        # contact_point_sphere_lis
        camera_info_path = color_path.parent.parent / \
            'camera_info' / color_path.with_suffix('.yaml').name
        depth_path = color_path.parent.parent / \
            'depth' / color_path.with_suffix('.npy').name

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

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsics)
        trimesh_pc = trimesh.PointCloud(
            np.asarray(
                pcd.points), np.asarray(
                pcd.colors))
        pc = skrobot.model.PointCloudLink(trimesh_pc)

        # visualzie groudv truth hangign point
        idx = int(regex.findall(color_path.stem)[0])
        # if idx == 0:
        #     continue
        category = color_path.parent.parent.name
        pose_file = osp.join(annotation_dir, category + '_{}.json'.format(idx))
        contact_points_dict = load_json(str(pose_file))
        contact_points = contact_points_dict['contact_points']
        contact_point_sphere_list = []
        for i, cp in enumerate(contact_points):
            # contact_point_sphere = skrobot.model.sphere(0.005, color=[255, 0, 0])
            contact_point_sphere = skrobot.model.Axis(0.003, 0.05)
            contact_point_sphere.newcoords(
                skrobot.coordinates.Coordinates(pos=cp[0], rot=cp[1:]))
            # contact_point_sphere.newcoords(
            # skrobot.coordinates.Coordinates(pos=cp[0],
            # rot=cp[1:]).rotate(np.pi, 'y'))
            contact_point_sphere_list.append(contact_point_sphere)

        # import ipdb
        # ipdb.set_trace()
        # visualize heatmap
        confidence = np.zeros(cv_bgr.shape[:2], dtype=np.uint32)
        for cp in contact_point_sphere_list:
            px, py = camera_model.project3d_to_pixel(cp.worldpos())
            create_gradient_circle(
                confidence, py, px)
        confidence = (confidence / confidence.max() * 255).astype(np.uint8)

        heatmap = overlay_heatmap(cv_bgr, confidence.astype(np.uint8))
        depth_bgr = colorize_depth(cv_depth, ignore_value=0, max_value=440)

        if gui:

            viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 640))
            viewer.add(pc)
            viewer.add(pc)
            for contact_point_sphere in contact_point_sphere_list:
                viewer.add(contact_point_sphere)
            viewer._init_and_start_app()

            cv2.imshow('heatmp', heatmap)
            cv2.imshow('depth', depth_bgr)
            cv2.waitKey()
            skip = False
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
