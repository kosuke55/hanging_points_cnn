# flake8: noqa

import argparse
from pathlib import Path

import numpy as np

from hanging_points_generator.generator_utils import load_json

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--gan-diff-directory', '-g', type=str,
    help='gan trained model diff directory generated by eval_ycb.py',
    # default='/media/kosuke55/SANDISK-2/meshdata/ycb_eval/eval_gan/diff') # sim hanging
    # default='/home/kosuke55/catkin_ws/src/hanging_points_cnn/data/ycb_real_eval/eval_gan/diff' # real hanging
    # default='/media/kosuke55/SANDISK-2/meshdata/ycb_sim_eval_pouring/eval_gan/diff') # sim pouring
    default='/home/kosuke55/catkin_ws/src/hanging_points_cnn/data/ycb_real_eval_pouring/eval_gan/diff') # real pouring
parser.add_argument(
    '--shapenet-diff-directory', '-s', type=str,
    help='shapenet trained model diff directory generated by eval_ycb.py',
    # default='/media/kosuke55/SANDISK-2/meshdata/ycb_eval/eval_gan/diff')  # sim hanging
    # default='/home/kosuke55/catkin_ws/src/hanging_points_cnn/data/ycb_real_eval/eval_shapnet/diff')  # real hanging
    # default='/media/kosuke55/SANDISK-2/meshdata/ycb_sim_eval_pouring/eval_shapenet/diff') # sim pouring
    default='/home/kosuke55/catkin_ws/src/hanging_points_cnn/data/ycb_real_eval_pouring/eval_shapenet/diff') # real pouring

args = parser.parse_args()

# gan
base_dir = args.gan_diff_directory
paths = list(sorted(Path(base_dir).glob('*.json')))
path = paths[1]
data = load_json(str(path))

result = {}

# only for hanging
two_labels_category = ['037_scissors',
                       '048_hammer']

one_tp_par_one_data = True

total = {'tp': 0, 'fp': 0, 'fn': 0,
         'pos_diff': [],
         'angle': []}
s_total = {'tp': 0, 'fp': 0, 'fn': 0,
           'pos_diff': [],
           'angle': []}

for path in paths:
    data = load_json(str(path))
    noise_num = 0
    category = path.stem[:-7]
    labels = [0, 1] if category in two_labels_category else [0]

    for label in labels:
        category = path.stem[:-7]

        if category in two_labels_category:
            category = category + '_' + str(label)

        if category not in result:
            result[category] = {
                'tp': [],
                'fn': [],
                'fp': [],
                'pos_diff': [],
                'angle': []}

        idx = int(path.stem[-6:])
        if data['-1']['angle'] != []:
            result[category]['fp'].append(idx)

        if str(label) not in data:
            result[category]['fn'].append(idx)
            continue

        if data[str(label)]['distance'] == []:
            print('fn', idx)
            result[category]['fn'].append(idx)
            continue

        for pos_diff, angle in zip(
                data[str(label)]['pos_diff'], data[str(label)]['angle']):
            result[category]['pos_diff'].append(pos_diff)
            result[category]['angle'].append(angle)
            total['pos_diff'].append(pos_diff)
            total['angle'].append(angle)
            if one_tp_par_one_data:
                if idx not in result[category]['tp']:
                    result[category]['tp'].append(idx)
            else:
                result[category]['tp'].append(idx)

for key in result.keys():
    result[key]['pos_mean'] = (np.mean(np.abs(result[key]['pos_diff']), axis=0) * 1000).tolist()
    result[key]['angle_mean'] = float(np.mean(result[key]['angle'])) / np.pi * 180


# shapenet
base_dir = args.shapenet_diff_directory
paths = list(sorted(Path(base_dir).glob('*.json')))
path = paths[1]
s_data = load_json(str(path))

shapenet_result = {}

for path in paths:
    s_data = load_json(str(path))
    noise_num = 0
    category = path.stem[:-7]
    labels = [0, 1] if category in two_labels_category else [0]

    for label in labels:
        category = path.stem[:-7]

        if category in two_labels_category:
            category = category + '_' + str(label)
        if category not in shapenet_result:
            shapenet_result[category] = {
                'tp': [],
                'fn': [],
                'fp': [],
                'pos_diff': [],
                'angle': []}

        idx = int(path.stem[-6:])
        if s_data['-1']['angle'] != []:
            shapenet_result[category]['fp'].append(idx)

        if str(label) not in s_data:
            print('fn', idx)
            shapenet_result[category]['fn'].append(idx)
            continue

        if s_data[str(label)]['distance'] == []:
            print('fn', idx)
            shapenet_result[category]['fn'].append(idx)
            continue

        for pos_diff, angle in zip(
                s_data[str(label)]['pos_diff'], s_data[str(label)]['angle']):
            shapenet_result[category]['pos_diff'].append(pos_diff)
            shapenet_result[category]['angle'].append(angle)
            s_total['pos_diff'].append(pos_diff)
            s_total['angle'].append(angle)
            if one_tp_par_one_data:
                if idx not in shapenet_result[category]['tp']:
                    print('tp', idx)
                    shapenet_result[category]['tp'].append(idx)
            else:
                shapenet_result[category]['tp'].append(idx)

for key in shapenet_result.keys():
    shapenet_result[key]['pos_mean'] = (np.mean(np.abs(shapenet_result[key]['pos_diff']), axis=0) * 1000).tolist()
    shapenet_result[key]['angle_mean'] = float(np.mean(shapenet_result[key]['angle'])) / np.pi * 180

for key in sorted(result.keys()):
    data = result[key]
    s_data = shapenet_result[key]

    for _key in ['tp', 'fp', 'fn']:
        total[_key] += len(data[_key])
        s_total[_key] += len(s_data[_key])

    if s_data['angle'] == [] and data['angle'] == []:
        print('{}  &{} &{} &{} &- &- &- &-  &{} &{} &{} &- &- &- &- \\\\'.format(
            key[4:].replace('_',' '),
            len(s_data['tp']),
            len(s_data['fp']), len(s_data['fn']),
            len(data['tp']),
            len(data['fp']), len(data['fn'])))
    elif s_data['angle'] == [] and data['angle'] != []:
        print('{}  &{} &{} &{} &- &- &- &-  &{} &{} &{} &{:.2f} &{:.2f} &{:.2f} &{:.2f} \\\\'.format(
            key[4:].replace('_',' '),
            len(s_data['tp']),
            len(s_data['fp']), len(s_data['fn']),
            len(data['tp']),
            len(data['fp']), len(data['fn']), data['pos_mean'][0], data['pos_mean'][1], data['pos_mean'][2], data['angle_mean']))
    elif s_data['angle'] != [] and data['angle'] == []:
        print('{}  &{} &{} &{} &{:.2f} &{:.2f} &{:.2f} &{:.2f}  &{} &{} &{} &- &- &- &- \\\\'.format(
            key[4:].replace('_',' '),
            len(s_data['tp']),
            len(s_data['fp']), len(s_data['fn']), s_data['pos_mean'][0], s_data['pos_mean'][1], s_data['pos_mean'][2], s_data['angle_mean'],
            len(data['tp']),
            len(data['fp']), len(data['fn'])))
    else:
        print('{}  &{} &{} &{} &{:.2f} &{:.2f} &{:.2f} &{:.2f}  &{} &{} &{} &{:.2f} &{:.2f} &{:.2f} &{:.2f} \\\\'.format(
            key[4:].replace('_',' '),
            len(s_data['tp']),
            len(s_data['fp']), len(s_data['fn']), s_data['pos_mean'][0], s_data['pos_mean'][1], s_data['pos_mean'][2], s_data['angle_mean'],
            len(data['tp']),
            len(data['fp']), len(data['fn']), data['pos_mean'][0], data['pos_mean'][1], data['pos_mean'][2], data['angle_mean']))

total['pos_mean'] = (np.mean(np.abs(total['pos_diff']), axis=0) * 1000).tolist()
total['angle_mean'] = float(np.mean(total['angle'])) / np.pi * 180
s_total['pos_mean'] = (np.mean(np.abs(s_total['pos_diff']), axis=0) * 1000).tolist()
s_total['angle_mean'] = float(np.mean(s_total['angle'])) / np.pi * 180

print('{}  &{} &{} &{} &{:.2f} &{:.2f} &{:.2f} &{:.2f}  &{} &{} &{} &{:.2f} &{:.2f} &{:.2f} &{:.2f} \\\\'.format(
    'total',
    s_total['tp'], s_total['fp'], s_total['fn'], s_total['pos_mean'][0], s_total['pos_mean'][1], s_total['pos_mean'][2], s_total['angle_mean'],
    total['tp'], total['fp'], total['fn'], total['pos_mean'][0], total['pos_mean'][1], total['pos_mean'][2], total['angle_mean']))
