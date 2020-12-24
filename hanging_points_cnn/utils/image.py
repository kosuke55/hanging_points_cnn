#!/usr/bin/env python3
# coding: utf-8

import copy
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
    normalized_depth = copy.copy(depth)
    if isinstance(depth, np.ndarray):
        min_value = np.nanmin(depth) if min_value is None else min_value
        max_value = np.nanmax(depth) if max_value is None else max_value
        remove_nan(normalized_depth)
        normalized_depth = (normalized_depth - min_value) / \
            (max_value - min_value)
        normalized_depth[normalized_depth <= 0] = 0
        normalized_depth[normalized_depth > 1] = 1
    else:
        normalized_depth = (normalized_depth - min_value) / \
            (max_value - min_value)

    return normalized_depth


def unnormalize_depth(normalized_depth, min_value, max_value):
    depth = copy.copy(normalized_depth)
    if isinstance(depth, np.ndarray):
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


def create_gradient_circle(img, cy, cx, sig=50., gain=100.):
    h, w = img.shape
    Y, X = np.ogrid[:h, :w]
    g = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2. * sig)) * gain
    img += g.astype(np.uint32)


def draw_axis(img, R, t, K, axis_length=0.1, copy=False):
    if copy:
        img = img.copy()

    rotV, _ = cv2.Rodrigues(R)
    points = np.vstack((
        np.eye(3) * axis_length, np.zeros((1, 3)))).astype(np.float32)
    axis_points, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))

    for color, axis_point in zip(
            [(255, 0, 0), (0, 255, 0), (0, 0, 255)], axis_points[:3][::-1]):
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


def draw_roi(img, roi, val=None, gt=False):
    if gt:
        color = (0, 0, 255)
        offset = 60
        width = 2
    else:
        color = (0, 255, 0)
        offset = 30
        width = 3

    img = img.copy()
    img = cv2.rectangle(
        img, (int(roi[0]), int(roi[1])),
        (int(roi[2]), int(roi[3])),
        color, width)
    if val is not None:
        cv2.putText(
            img, str(round(val, 3)),
            (int(roi[0]), int(roi[1] + offset)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
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
            shape, resample=interpolation))


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


def get_gradation(
        width, height, max_value, horizontal=True):
    """Get gradation image for depth deformation

    Parameters
    ----------
    width : int
        image width
    height : int
        image height
    max_value : float
        max of random value
    horizontal : bool, optional
        If True make horizontal gradatation else vetical gradation,
        by default True

    Returns
    -------
    gradation : numpy.ndarray
        gradation image
    """
    start_value = max_value * np.random.rand()
    end_value = -start_value
    if horizontal:
        return np.tile(np.linspace(
            start_value, end_value, width), (height, 1))
    else:
        return np.tile(np.linspace(
            start_value, end_value, height), (width, 1)).T


def depth_to_mask(depth):
    """Create mask image from depth

    Parameters
    ----------
    depth : numpy.ndarray

    Returns
    -------
    mask : numpy.ndarray
    """
    mask = np.zeros_like(depth, np.uint8)
    mask[depth > 0] = 255
    return mask


def mask_to_edges(mask, kenel=(5, 5)):
    """Convert mask to edges

    Parameters
    ----------
    mask : numpy.ndarray
    kenel : tuple, optional
        dilate kernel, by default (5, 5)

    Returns
    -------
    edges : numpy.ndarray
    """
    edges = cv2.Canny(mask, 0, 255)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def depth_edges_noise(depth, value=5., copy=True):
    """Add noise to depth edges

    Parameters
    ----------
    depth : numpy.ndarray
    value : float, optional
        noise distacne value, by default 5.
    copy : bool, optional
        by default True

    Returns
    -------
    depth : numpy.ndarray
        depth with noise
    """
    if copy:
        depth = depth.copy()
    mask = depth_to_mask(depth)
    edges = mask_to_edges(mask)
    noise = (np.random.random_sample(edges.shape) - 0.5) * 2 * value
    noise[edges == 0] = 0
    noise[mask == 0] = 0
    depth += noise
    return depth


def depth_edges_erase(depth, max_sample=100, copy=True):
    """Erase depth edge

    Parameters
    ----------
    depth : numpy.ndarray
    max_sample : int, optional
        max value of sample, by default 100
    copy : bool, optional
        by default True

    Returns
    -------
    depth : numpy.ndarray
        Depth with randomly erased edges
    """
    if copy:
        depth = depth.copy()
    mask = depth_to_mask(depth)
    edges = mask_to_edges(mask)
    edges_idx = np.where(edges == 255)
    length = len(edges_idx[0])
    num_samples = np.random.randint(0, min(max_sample, length))
    samples_idx = np.unique(np.random.randint(0, length, num_samples))
    py_list = edges_idx[0][samples_idx]
    px_list = edges_idx[1][samples_idx]

    erase_mask = np.zeros_like(mask, dtype=np.uint8)
    for i in range(len(px_list)):
        _erase_mask = create_circular_mask(
            erase_mask.shape[0], erase_mask.shape[1],
            py_list[i], px_list[i], radius=np.random.randint(1, 10))
        erase_mask[_erase_mask] = 255

    depth[erase_mask == 255] = 0
    return depth


def overlay_heatmap(image, heatmap):
    """Overlay heatmap on image

    Parameters
    ----------
    image : numpy.ndarray
    heatmap : numpy.ndarray

    Returns
    -------
    added_image : numpy.ndarray
    """
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    added_image = cv2.addWeighted(image, 0.3, heatmap, 0.7, 0)
    return added_image
