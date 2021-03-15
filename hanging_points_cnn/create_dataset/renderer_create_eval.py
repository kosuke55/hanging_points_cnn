#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import argparse
import os.path as osp
from pathlib import Path
import sys

import numpy as np

from renderer import Renderer
from renderer import get_contact_points
from renderer import make_save_dirs
from renderer import sample_contact_points
from renderer import split_file_name


if __name__ == '__main__':
    print('Start')
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--save-dir', '-s',
        type=str, help='save dir',
        default='/media/kosuke/SANDISK-2/meshdata/ycb_sim_eval_hanging')
        # default='/media/kosuke/SANDISK-2/meshdata/ycb_sim_eval_pouring')
    parser.add_argument(
        '--input-dir', '-i',
        type=str, help='input directory',
        # default='/media/kosuke55/SANDISK/meshdata/ycb_hanging_object/urdf_eval')
        default='/media/kosuke55/SANDISK/meshdata/ycb_pouring_object_16/textured_urdf')
    parser.add_argument(
        '--annotation-dir', '-a',
        type=str, help='annotation directory',
        # default='/media/kosuke55/SANDISK/meshdata/ycb_hanging_object/urdf2/annotation_obj_1027')
        default='/media/kosuke55/SANDISK/meshdata/ycb_pouring_object_16/textured_urdf/annotation_obj')
    parser.add_argument(
        '--dataset-type', '-dt',
        type=str, help='dataset type',
        default='')
    parser.add_argument(
        '--urdf-name', '-u',
        type=str, help='save dir',
        default='textured.urdf')
    parser.add_argument(
        '--skip-list', '-sl',
        type=str, help='skip list file name',
        default='filter_skip_list.txt')
    parser.add_argument(
        '--gui', '-g',
        action='store_true', help='debug gui')
    parser.add_argument(
        '--task', '-t', type=str,
        help='h(hanging) or p(pouring)',
        default='h')
    parser.add_argument(
        '--interactive',
        action='store_true', help='interactive save mode')
    args = parser.parse_args()

    dataset_type = args.dataset_type
    gui = args.gui
    input_dir = args.input_dir
    annotation_dir = args.annotation_dir
    save_dir_base = args.save_dir
    urdf_name = args.urdf_name
    skip_list = args.skip_list
    interctive = args.interactive

    task_type = args.task
    if task_type == 'p':
        task_type = 'pouring'
    else:
        task_type = 'hanging'
    print('task type: {}'.format(task_type))

    file_paths = list(sorted(Path(
        input_dir).glob(osp.join('*', urdf_name))))
    annotation_path = Path(annotation_dir)

    num_scene = 12
    required_data = 5

    if task_type == 'hanging':
        category_name_list = [
            "019_pitcher_base",
            "022_windex_bottle",
            "025_mug",
            "033_spatula", # no contact pointsa
            "035_power_drill",
            "037_scissors",
            "042_adjustable_wrench",
            "048_hammer",
            "050_medium_clamp",
            "051_large_clamp",
            "052_extra_large_clamp"
        ]

        obj_rot_list = [
            [0, 0, 0],  # 019_pitcher_base
            [0, 0, 0],  # 022_windex_bottle
            [0, 0, 0],  # 025_mug
            [0, np.pi / 2, 0],  # 033_spatula
            [0, 0, np.pi / 2],  # 035_power_drill
            [0, 0, -np.pi / 2],  # 037_scissors
            [0, 0, np.pi / 2],  # 042_adjustable_wrench
            [0, 0, np.pi / 2],  # 048_hammer
            [0, np.pi / 2, 0],  # 050_medium_clamp
            [0, 0, np.pi / 2],  # 051_large_clamp
            [0, np.pi / 2, 0],  # 052_extra_large_clamp
        ]

        radius = 0.5
        camera_z = 0.3
    else:
        category_name_list = [
            "024_bowl",
            "025_mug",
            "027_skillet",
            "029_plate"
        ]

        obj_rot_list = [
            [0, 0, 0],  # 024_bowl
            [0, 0, 0],  # 025_mug
            [0, 0, 0],  # 027_skillet"
            [0, 0, 0]   # 029_plate
        ]

        radius = 0.3
        camera_z = 0.5

    selected_category_file_paths = []
    for file_path in file_paths:
        if any(c in str(file_path) for c in category_name_list):
            selected_category_file_paths.append(file_path)
    file_paths = selected_category_file_paths

    try:
        for file_path, obj_rot in zip(file_paths, obj_rot_list):
            dirname, filename, category_name, idx \
                = split_file_name(str(file_path), dataset_type)
            print('category_name: ' + category_name)

            if category_name_list is not None:
                if category_name not in category_name_list:
                    continue

            contact_points, labels = get_contact_points(
                str(annotation_path / Path(
                    category_name).with_suffix('.json')),
                use_clustering=False, use_filter_penetration=False,
                inf_penetration_check=False, get_label=True)
            print('Load contact points: {}'.format(len(contact_points)))
            if contact_points is None:
                continue

            save_dir = osp.join(save_dir_base, category_name)
            save_dir = make_save_dirs(save_dir)

            contact_points, idx = sample_contact_points(
                contact_points, 100, get_idx=True)
            labels = [labels[i] for i in idx]
            print('Sample contact points: {}'.format(len(contact_points)))

            save_data = 0
            i = 0
            while save_data < required_data:
                if i == num_scene:
                    i = 0
                r = Renderer(
                    gui=gui, save_dir=save_dir,
                    use_change_light=False, labels=labels)
                camera_pos = [
                    np.cos(
                        i % num_scene * np.pi * 2 / num_scene) * radius,
                    np.sin(
                        i % num_scene * np.pi * 2 / num_scene) * radius,
                    camera_z]
                if r.create_eval_data(
                    osp.join(dirname, urdf_name),
                    contact_points, camera_pos,
                    obj_pos=[0, 0, 0], obj_rot=obj_rot,
                        gui=gui, interactive=interctive):
                    save_data += 1
                print(r.data_id)
                i += 1
                if save_data == required_data:
                    break

    except KeyboardInterrupt:
        sys.exit()
