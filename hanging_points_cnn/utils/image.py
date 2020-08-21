#!/usr/bin/env python3
# coding: utf-8

import sys
from PIL import Image

import numpy as np

try:
    import cv2
except ImportError:
    for path in sys.path:
        if '/opt/ros/' in path:
            print('sys.path.remove({})'.format(path))
            sys.path.remove(path)
            import cv2
            sys.path.append(path)


def get_depth_in_roi(depth, roi, depth_range=None):
    """Get median depth in roi

    Parameters
    ----------
    depth : np.ndarray
        depth image
    roi : list[int]
        [xmin, ymin, xmax, ymax]
    """
    depth = depth.copy()
    depth = depth[int(roi[1]):int(roi[3]), int(roi[0]):int(roi[2])]
    if depth_range is not None:
        depth = depth[np.where(np.logical_and(
            depth_range[0] < depth, depth < depth_range[1]))]
    depth = np.median(depth)

    return depth


def remove_nan(img):
    nan_mask = np.isnan(img)
    img[nan_mask] = 0


def normalize_depth(depth, min_value=None, max_value=None):
    min_value = np.nanmin(depth) if min_value is None else min_value
    max_value = np.nanmax(depth) if max_value is None else max_value
    normalized_depth = depth.copy()
    remove_nan(normalized_depth)
    normalized_depth = (normalized_depth - min_value) / (max_value - min_value)
    normalized_depth[normalized_depth <= 0] = 0
    normalized_depth[normalized_depth > 1] = 1

    return normalized_depth


def unnormalize_depth(normalized_depth, min_value, max_value):
    depth = normalized_depth.copy()
    remove_nan(depth)
    depth = depth * (max_value - min_value) + min_value

    return depth


def colorize_depth(depth, min_value=None, max_value=None, ignore_value=None):
    # if ignore_zero and min_value is None:
    depth = depth.copy()
    if ignore_value is not None:
        depth[depth == ignore_value] = np.nan

    normalized_depth = normalize_depth(depth, min_value, max_value)
    nan_mask = np.isnan(normalized_depth)
    gray_depth = normalized_depth * 255
    gray_depth = gray_depth.astype(np.uint8)
    colorized = cv2.applyColorMap(gray_depth, cv2.COLORMAP_JET)
    colorized[nan_mask] = (0, 0, 0)

    return colorized


def create_circular_mask(h, w, cy, cx, radius=50):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = dist_from_center <= radius
    return mask


def create_depth_circle(img, cy, cx, value, radius=50, copy=True):
    if copy:
        img = img.copy()
    depth_mask = np.zeros_like(img)
    depth_mask[np.where(img == 0)] = 1
    circlular_mask = np.zeros_like(img)
    circlular_mask_idx = np.where(
        create_circular_mask(img.shape[0], img.shape[1], cy, cx,
                             radius=radius))
    circlular_mask[circlular_mask_idx] = 1
    img[circlular_mask_idx] = value
    return img


def draw_axis(img, R, t, K, axis_length=0.1, copy=False):
    if copy:
        img = img.copy()

    rotV, _ = cv2.Rodrigues(R)
    points = np.vstack((
        np.eye(3) * axis_length, np.zeros((1, 3)))).astype(np.float32)
    axis_points, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))

    for color, axis_point in zip(
            [(0, 0, 255), (0, 255, 0), (255, 0, 0)], axis_points):
        img = cv2.line(
            img, tuple(axis_points[3].ravel()), tuple(axis_point.ravel()),
            color, 2)

    return img


def draw_vec(img, vec, t, cm, vec_length=0.1, copy=False):
    if copy:
        img = img.copy()

    s = cm.project3d_to_pixel(t)
    e = cm.project3d_to_pixel(t + vec)
    img = cv2.line(
        img, s, e,
        (0, 0, 255), 2)

    return img


def resize_np_img(
        img, shape, interpolation=Image.BILINEAR):
    """Resize numpy image.

    Parameters
    ----------
    img : numpy.ndarray
        numpy image
    shape : tuple
        (width, height) order. Note numpy image shape is (height ,width)

    interpolation : int
        interpolation method.
        You can specify, PIL.Image.NEAREST, PIL.Image.BILINEAR,
        PIL.Image.BICUBIC and PIL.Image.LANCZOS.

    Returns
    -------
    reshaped np_image

    """
    return np.array(
        Image.fromarray(img).resize(
            shape, interpolation=interpolation))


def trim_depth(dep, depth):
    """Get the trimmed detph value

    Parameters
    ----------
    dep : float
        depth value
    depth : numpy.ndarray
        depth image

    Returns
    -------
    trimmed_dep
        depth value which is trimmed by depth image
    """
    dep = np.nanmin(depth) if dep < np.nanmin(depth) else dep
    dep = np.nanmax(depth) if np.nanmax(depth) < dep else dep
    return dep
