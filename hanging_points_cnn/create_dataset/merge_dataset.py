#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import os
import os.path as osp
import subprocess


def merge_two_dataset(input_dir_1, input_dir_2, out_dir):
    '''
    Merge input_dir_1 and input_dir_2 into out_dir.
    '''

    os.makedirs(out_dir, exist_ok=True)

    if input_dir_1 != out_dir:
        cmd = 'sudo cp -r ' + input_dir_1 + '/clip_info ' + out_dir
        print(cmd)
        subprocess.call(
            [cmd],
            shell=True)
        cmd = 'sudo cp -r ' + input_dir_1 + '/debug_axis ' + out_dir
        print(cmd)
        subprocess.call(
            [cmd],
            shell=True)
        cmd = 'sudo cp -r ' + input_dir_1 + '/depth ' + out_dir
        print(cmd)
        subprocess.call(
            [cmd],
            shell=True)
        cmd = 'sudo cp -r ' + input_dir_1 + \
            '/hanging_points_depth ' + out_dir
        print(cmd)
        subprocess.call(
            [cmd],
            shell=True)
        cmd = 'sudo cp -r ' + input_dir_1 + '/heatmap ' + out_dir
        print(cmd)
        subprocess.call(
            [cmd],
            shell=True)
        cmd = 'sudo cp -r ' + input_dir_1 + '/rotations ' + out_dir
        print(cmd)
        subprocess.call(
            [cmd],
            shell=True)
        cmd = 'sudo cp -r ' + input_dir_1 + '/intrinsics ' + out_dir
        print(cmd)
        subprocess.call(
            [cmd],
            shell=True)

    files_1 = glob.glob(osp.join(input_dir_1, 'depth', '*'))
    files_1.sort()

    dirname, filename = osp.split(files_1[len(files_1) - 1])

    files_1_last_idx, ext = osp.splitext(filename)
    files_1_last_idx = int(files_1_last_idx)

    files_2 = glob.glob(osp.join(input_dir_2, 'depth', '*'))
    # Empty dir
    if files_2 == []:
        return
    files_2.sort()

    try:
        dirname, filename = osp.split(files_2[len(files_2) - 1])
    except:
        import ipdb
        ipdb.set_trace()
    files_2_last_idx, ext = osp.splitext(filename)
    files_2_last_idx = int(files_2_last_idx)

    for idx in range(files_2_last_idx + 1):
        cmd = 'sudo cp ' + input_dir_2 + '/clip_info/{:06}.npy '.format(idx) + osp.join(
            out_dir, 'clip_info', '{:06}.npy'.format(files_1_last_idx + idx + 1))
        print(cmd)
        subprocess.call(
            [cmd],
            shell=True)
        cmd = 'sudo cp ' + input_dir_2 + '/debug_axis/{:06}.png '.format(idx) + osp.join(
            out_dir, 'debug_axis', '{:06}.png'.format(files_1_last_idx + idx + 1))
        print(cmd)
        subprocess.call(
            [cmd],
            shell=True)
        cmd = 'sudo cp ' + input_dir_2 + '/depth/{:06}.npy '.format(idx) + osp.join(
            out_dir, 'depth', '{:06}.npy'.format(files_1_last_idx + idx + 1))
        print(cmd)
        subprocess.call(
            [cmd],
            shell=True)
        cmd = 'sudo cp ' + input_dir_2 + '/hanging_points_depth/{:06}.npy '.format(idx) + osp.join(
            out_dir, 'hanging_points_depth', '{:06}.npy'.format(files_1_last_idx + idx + 1))
        print(cmd)
        subprocess.call(
            [cmd],
            shell=True)
        cmd = 'sudo cp ' + input_dir_2 + '/heatmap/{:06}.png '.format(idx) + osp.join(
            out_dir, 'heatmap', '{:06}.png'.format(files_1_last_idx + idx + 1))
        print(cmd)
        subprocess.call(
            [cmd],
            shell=True)
        cmd = 'sudo cp ' + input_dir_2 + '/rotations/{:06}.npy '.format(idx) + osp.join(
            out_dir, 'rotations', '{:06}.npy'.format(files_1_last_idx + idx + 1))
        print(cmd)
        subprocess.call(
            [cmd],
            shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_dir_1',
        '-i1',
        type=str,
        help='input dir 1',
        default='/media/kosuke/SANDISK/meshdata/ycb_hanging_object/hoge/kosuke-mouse-2020-06-18-14-23-55-721091-5714')
    parser.add_argument(
        '--input_dir_2',
        '-i2',
        type=str,
        help='input dir 2',
        default='/media/kosuke/SANDISK/meshdata/ycb_hanging_object/hoge/kosuke-mouse-2020-06-18-14-23-57-348757-5757')
    parser.add_argument(
        '--out_dir',
        '-o',
        type=str,
        help='out dir',
        default='/ media/kosuke/SANDISK/meshdata/ycb_hanging_object/hoge/merged
    args=parser.parse_args()

    input_dir_1=args.input_dir_1
    input_dir_2=args.input_dir_2
    out_dir=args.out_dir

    merge_two_dataset(input_dir_1, input_dir_2, out_dir)
