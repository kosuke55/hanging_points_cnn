import numpy as np
from skrobot.coordinates.math import quaternion2matrix


def two_vectors_angle(v1, v2):
    """Calculate the angle between two vectors

    Parameters
    ----------
    v1 : numpy.ndarray
        [x, y, z] order
    v2 : numpy.ndarray
        [x, y, z] order

    Returns
    -------
    angle : float
    """
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(cos)


def quaternion2vec(q, axis='x'):
    """Calculate axis vector from quaternion

    Parameters
    ----------
    q : np.ndarray
        qauternion
    axis : str, optional
        axis x, y, z, by default 'x'

    Returns
    -------
    vec : numpy.ndarray
    """
    m = quaternion2matrix(q)
    return matrix2vec(m, axis=axis)


def matrix2vec(m, axis='x'):
    """Calculate axis vector from rotation matrix

    Parameters
    ----------
    m : numpy.ndarray
        rotation matrix
    axis : str, optional
        axis x, y, z, by default 'x'

    Returns
    -------
    vec : numpy.ndarray

    Raises
    ------
    ValueError
        axis shoule be x, y, z
    """
    if axis == 'x':
        vec = m[:, 0]
    elif axis == 'y':
        vec = m[:, 1]
    elif axis == 'z':
        vec = m[:, 2]
    else:
        raise ValueError("Valid axis are 'x', 'y', 'z'")
    return vec
