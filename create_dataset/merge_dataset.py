#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import os
import os.path as osp
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_dir_1',
        '-i1',
        type=str,
        help='input dir 1',
        default='/media/kosuke/SANDISK/meshdata/Hanging-ObjectNet3D-DoubleFaces/hoge')
    # default='/media/kosuke/SANDISK/meshdata/ycb_hanging_object/0603')
    parser.add_argument(
        '--input_dir_2',
        '-i2',
        type=str,
        help='input dir 2',
        default='/media/kosuke/SANDISK/meshdata/Hanging-ObjectNet3D-DoubleFaces/hoge')
    # default='/media/kosuke/SANDISK/meshdata/Hanging-ObjectNet3D-DoubleFaces/cup_key_scissors_0528')
    parser.add_argument(
        '--out_dir',
        '-o',
        type=str,
        help='out dir',
        default='/media/kosuke/SANDISK/meshdata/merged')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    subprocess.call(
        ['sudo cp -r ' + args.input_dir_1 + '/clip_info ' + args.out_dir],
        shell=True)
    subprocess.call(
        ['sudo cp -r ' + args.input_dir_1 + '/debug_axis ' + args.out_dir],
        shell=True)
    subprocess.call(
        ['sudo cp -r ' + args.input_dir_1 + '/depth ' + args.out_dir],
        shell=True)
    subprocess.call(
        ['sudo cp -r ' + args.input_dir_1 + '/heatmap ' + args.out_dir],
        shell=True)
    subprocess.call(
        ['sudo cp -r ' + args.input_dir_1 + '/rotations ' + args.out_dir],
        shell=True)

    files_1 = glob.glob(osp.join(args.input_dir_1, 'color', '*'))
    files_1.sort()
    dirname, filename = osp.split(files_1[len(files_1) - 1])
    files_1_last_idx, ext = osp.splitext(filename)
    files_1_last_idx = int(files_1_last_idx)

    files_2 = glob.glob(osp.join(args.input_dir_2, 'color', '*'))
    files_2.sort()
    dirname, filename = osp.split(files_2[len(files_2) - 1])
    files_2_last_idx, ext = osp.splitext(filename)
    files_2_last_idx = int(files_2_last_idx)

    for idx in range(files_2_last_idx):
        cmd = 'sudo cp ' + args.input_dir_1 + '/clip_info/{:05}.npy '.format(idx) + osp.join(
            args.out_dir, 'clip_info', '{:05}.npy'.format(files_1_last_idx + idx + 1))
        print(cmd)
        subprocess.call(
            [cmd],
            shell=True)
        cmd = 'sudo cp ' + args.input_dir_1 + 'debug_axis/{:05}.png '.format(idx) + osp.join(
            args.out_dir, 'debug_axis', '{:05}.png'.format(files_1_last_idx + idx + 1))
        print(cmd)
        subprocess.call(
            [cmd],
            shell=True)
        cmd = 'sudo cp ' + args.input_dir_1 + '/depth/{:05}.npy '.format(idx) + osp.join(
            args.out_dir, 'depth', '{:05}.npy'.format(files_1_last_idx + idx + 1))
        print(cmd)
        subprocess.call(
            [cmd],
            shell=True)
        cmd = 'sudo cp ' + args.input_dir_1 + '/heatmap/{:05}.png '.format(idx) + osp.join(
            args.out_dir, 'heatmap', '{:05}.png'.format(files_1_last_idx + idx + 1))
        print(cmd)
        subprocess.call(
            [cmd],
            shell=True)
        cmd = 'sudo cp ' + args.input_dir_1 + '/rotations/{:05}.npy '.format(idx) + osp.join(
            args.out_dir, 'rotations', '{:05}.npy'.format(files_1_last_idx + idx + 1))
        print(cmd)
        subprocess.call(
            [cmd],
            shell=True)
