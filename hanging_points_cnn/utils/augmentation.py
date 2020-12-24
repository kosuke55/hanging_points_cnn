#!/usr/bin/env python3
# coding: utf-8

import imgaug.augmenters as iaa  # for python3
from hanging_points_cnn.utils.image import depth_edges_erase
from hanging_points_cnn.utils.random_eraser import get_random_eraser


def augment_depth(depth, random_value=False):
    """data augmentation for depth image

    Parameters
    ----------
    depth : numpy.ndarray
    random_value : bool, optional
        if true, erase with pixel level random value,
        by default False

    Returns
    -------
    depth : numpy.ndarray
    """
    aug_seq = iaa.Sequential([
        iaa.Dropout([0, 0.8]),
        iaa.MultiplyElementwise((0.99, 1.01)),
    ], random_order=True)

    depth = depth_edges_erase(depth)
    depth = aug_seq.augment_image(depth)

    nonzero_depth = depth.copy()
    nonzero_depth[nonzero_depth == 0] = depth.max()

    if random_value:
        depth_eraser = get_random_eraser(
            p=0.9, s_l=0.1, s_h=0.5,
            v_l=nonzero_depth.min(),
            v_h=depth.max(),
            pixel_level=False)
    else:
        depth_eraser = get_random_eraser(
            p=0.9, s_l=0.1, s_h=0.5,
            v_l=0,
            v_h=0,
            pixel_level=False)
    depth = depth_eraser(depth)

    return depth
