# https://github.com/pytorch/pytorch/issues/40140
# torch.save(a, 'a.pt', _use_new_zipfile_serialization=False)

import argparse

import torch

from hanging_points_cnn.learning_scripts.hpnet import HPNET


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--pretrained_model',
        '-p',
        type=str,
        help='Pretrained models',
        required=True
        )  # noqa

    args = parser.parse_args()
    pretrained_model = args.pretrained_model

    config = {
        'output_channels': 1,
        'feature_extractor_name': 'resnet50',
        'confidence_thresh': 0.1,
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

    torch.save(model.state_dict(), pretrained_model,
               _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    main()
