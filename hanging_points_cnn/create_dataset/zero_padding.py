#!/usr/bin/env python
# coding: utf-8

'''
Rename file with zero padding.
1.npy -> 000001.npy
00001.npy -> 000001.npy
'''

import argparse
import glob
import os.path as osp
import subprocess


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--input_dir',
    '-i',
    type=str,
    help='input dir',
    required=True)


args = parser.parse_args()
input_dir = args.input_dir

files = glob.glob(osp.join(input_dir, '*/*'))

for file in files:
    dirname, filename = osp.split(file)
    idx, ext = osp.splitext(filename)
    try:
        idx = int(idx)
    except Exception:
        continue

    cmd = 'mv ' + file + ' ' + dirname + '/{:06}'.format(idx) + ext
    print(cmd)
    subprocess.call(
        [cmd],
        shell=True)
    print(idx)

