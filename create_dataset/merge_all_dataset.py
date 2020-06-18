#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import os.path as osp

from merge_dataset import merge_two_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--base_dir',
        '-b',
        type=str,
        help='base_dir',
        default='/media/kosuke/SANDISK/meshdata/ycb_hanging_object/hoge')
    parser.add_argument(
        '--out_dir',
        '-o',
        type=str,
        help='out dir',
        default='/media/kosuke/SANDISK/meshdata/ycb_hanging_object/hoge/merged')

    args = parser.parse_args()
    base_dir = args.base_dir
    out_dir = args.out_dir
    dirs = glob.glob(osp.join(base_dir, '*'))

    c = 0
    for dir in dirs:
        f = glob.glob(osp.join(dir, 'depth', '*'))
        # print(f)
        print(dir + ', len:{}'.format(len(f)))
        c += len(f)

    merge_two_dataset(input_dir_1=dirs[0],
                      input_dir_2=dirs[1], out_dir=out_dir)
    for i in range(len(dirs) - 2):
        print(dirs[i + 2])
        merge_two_dataset(input_dir_1=out_dir,
                          input_dir_2=dirs[i + 2], out_dir=out_dir)

    f = glob.glob(osp.join(out_dir, 'depth', '*'))

    print('Expected number: {}, Actual number: {}'.format(c, len(f)))
    assert c == len(f), '{} was expected but {}'.format(c, len(f))
