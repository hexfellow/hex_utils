#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-01-14
################################################################

import numpy as np
from typing import Tuple


def hat(vec: np.ndarray):
    """so(3) vector → skew-symmetric matrix"""
    assert vec.shape == (3, ), "cross_matrix vec shape err"

    trans = np.array([
        [0.0, -vec[2], vec[1]],
        [vec[2], 0.0, -vec[0]],
        [-vec[1], vec[0], 0.0],
    ])
    return trans


def vee(mat: np.ndarray):
    """skew-symmetric matrix → so(3) vector"""
    assert mat.shape == (3, 3), "cross_matrix_inv mat shape err"

    vec = np.array([mat[2, 1], mat[0, 2], mat[1, 0]])
    return vec


def rad2deg(rad):
    deg = rad * 180.0 / np.pi
    return deg


def deg2rad(deg):
    rad = deg * np.pi / 180.0
    return rad


def angle_norm(rad):
    normed_rad = (rad + np.pi) % (2 * np.pi) - np.pi
    return normed_rad


def quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    assert q1.shape == (4, ) and q2.shape == (4, ), "quat_slerp quat shape err"

    # normalize
    q1_norm = q1 / np.linalg.norm(q1)
    if q1_norm[0] < 0.0:
        q1_norm = -q1_norm
    q2_norm = q2 / np.linalg.norm(q2)
    if q2_norm[0] < 0.0:
        q2_norm = -q2_norm

    # dot
    dot = np.dot(q1_norm, q2_norm)
    if dot < 0.0:
        q2_norm = -q2_norm
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)

    # slerp
    if np.fabs(theta) < 1e-6:
        q = q1_norm + t * (q2_norm - q1_norm)
        q = q / np.linalg.norm(q)
        return q

    sin_theta = np.sin(theta)
    q1_factor = np.sin((1 - t) * theta) / sin_theta
    q2_factor = np.sin(t * theta) / sin_theta
    q = q1_factor * q1_norm + q2_factor * q2_norm
    return q


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    assert q1.shape == (4, ) and q2.shape == (4, ), "quat_mul quat shape err"

    # normalize
    q1_norm = q1 / np.linalg.norm(q1)
    if q1_norm[0] < 0.0:
        q1_norm = -q1_norm
    q2_norm = q2 / np.linalg.norm(q2)
    if q2_norm[0] < 0.0:
        q2_norm = -q2_norm

    # mul
    w1, x1, y1, z1 = q1_norm
    w2, x2, y2, z2 = q2_norm
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    q = np.array([w, x, y, z])
    return q


def quat_inv(quat: np.ndarray) -> np.ndarray:
    assert quat.shape == (4, ), "quat_inv quat shape err"
    q = quat / np.linalg.norm(quat)
    if q[0] < 0.0:
        q = -q

    # inv
    inv = np.array([q[0], -q[1], -q[2], -q[3]])
    return inv


def trans_inv(trans: np.ndarray) -> np.ndarray:
    assert trans.shape == (4, 4), "trans2part trans shape err"

    pos = trans[:3, 3]
    rot = trans[:3, :3]

    # inv
    inv = np.eye(4)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -inv[:3, :3] @ pos
    return inv


def rot2quat(rot: np.ndarray) -> np.ndarray:
    assert rot.shape == (3, 3), "rot2quat rot shape err"

    qw, qx, qy, qz = 1, 0, 0, 0
    trace = np.trace(rot)
    if trace > 0:
        temp = 2.0 * np.sqrt(1 + trace)
        qw = 0.25 * temp
        qx = (rot[2, 1] - rot[1, 2]) / temp
        qy = (rot[0, 2] - rot[2, 0]) / temp
        qz = (rot[1, 0] - rot[0, 1]) / temp
    else:
        if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            temp = 2.0 * np.sqrt(1 + rot[0, 0] - rot[1, 1] - rot[2, 2])
            qw = (rot[2, 1] - rot[1, 2]) / temp
            qx = 0.25 * temp
            qy = (rot[1, 0] + rot[0, 1]) / temp
            qz = (rot[0, 2] + rot[2, 0]) / temp
        elif rot[1, 1] > rot[2, 2]:
            temp = 2.0 * np.sqrt(1 + rot[1, 1] - rot[0, 0] - rot[2, 2])
            qw = (rot[0, 2] - rot[2, 0]) / temp
            qx = (rot[1, 0] + rot[0, 1]) / temp
            qy = 0.25 * temp
            qz = (rot[2, 1] + rot[1, 2]) / temp
        else:
            temp = 2.0 * np.sqrt(1 + rot[2, 2] - rot[0, 0] - rot[1, 1])
            qw = (rot[1, 0] - rot[0, 1]) / temp
            qx = (rot[0, 2] + rot[2, 0]) / temp
            qy = (rot[2, 1] + rot[1, 2]) / temp
            qz = 0.25 * temp

    return np.array([qw, qx, qy, qz])


def rot2axis(rot: np.ndarray) -> Tuple[np.ndarray, float]:
    assert rot.shape == (3, 3), "rot2axis rot shape err"

    cos_theta = 0.5 * (np.trace(rot) - 1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-6:
        return np.array([1.0, 0.0, 0.0]), 0.0
    else:
        axis_matrix = (rot - rot.T) / (2 * np.sin(theta))
        axis = vee(axis_matrix)
    return axis, theta


def rot2so3(rot: np.ndarray) -> np.ndarray:
    assert rot.shape == (3, 3), "rot2so3 rot shape err"

    axis, theta = rot2axis(rot)
    return theta * axis


def quat2rot(quat: np.ndarray) -> np.ndarray:
    assert quat.shape == (4, ), "quat2rot quat shape err"
    q = quat / np.linalg.norm(quat)
    if q[0] < 0.0:
        q = -q

    # temp vars
    qx2 = q[1] * q[1]
    qy2 = q[2] * q[2]
    qz2 = q[3] * q[3]
    qxqw = q[1] * q[0]
    qyqw = q[2] * q[0]
    qzqw = q[3] * q[0]
    qxqy = q[1] * q[2]
    qyqz = q[2] * q[3]
    qzqx = q[3] * q[1]

    # rot
    rot = np.array([
        [
            1 - 2 * (qy2 + qz2),
            2 * (qxqy - qzqw),
            2 * (qzqx + qyqw),
        ],
        [
            2 * (qxqy + qzqw),
            1 - 2 * (qx2 + qz2),
            2 * (qyqz - qxqw),
        ],
        [
            2 * (qzqx - qyqw),
            2 * (qyqz + qxqw),
            1 - 2 * (qx2 + qy2),
        ],
    ])
    return rot


def quat2axis(quat: np.ndarray) -> Tuple[np.ndarray, float]:
    assert quat.shape == (4, ), "quat2axis quat shape err"
    q = quat / np.linalg.norm(quat)
    if q[0] < 0.0:
        q = -q

    vec = q[1:]
    norm_vec = np.linalg.norm(vec)
    if norm_vec < 1e-6:
        return np.array([1.0, 0.0, 0.0]), 0.0

    theta = 2 * np.arctan2(norm_vec, q[0])
    axis = vec / norm_vec
    return axis, theta


def quat2so3(quat: np.ndarray) -> np.ndarray:
    assert quat.shape == (4, ), "quat2so3 quat shape err"

    axis, theta = quat2axis(quat)
    return theta * axis


def axis2rot(axis: np.ndarray, theta: float) -> np.ndarray:
    assert axis.shape == (3, ), "axis2rot axis shape err"
    if theta < 1e-6:
        return np.eye(3)

    axis_matrix = hat(axis)
    rot = np.eye(3) + np.sin(theta) * axis_matrix + (1 - np.cos(theta)) * (
        axis_matrix @ axis_matrix)
    return rot


def axis2quat(axis: np.ndarray, theta: float) -> np.ndarray:
    assert axis.shape == (3, ), "axis2quat axis shape err"
    if theta < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])

    quat = np.zeros(4)
    quat[0] = np.cos(theta / 2)
    quat[1:] = axis * np.sin(theta / 2)
    return quat


def axis2so3(axis: np.ndarray, theta: float) -> np.ndarray:
    assert axis.shape == (3, ), "axis2so3 axis shape err"

    return theta * axis


def so32rot(so3: np.ndarray) -> np.ndarray:
    assert so3.shape == (3, ), "so32quat so3 shape err"

    theta = np.linalg.norm(so3)
    if theta < 1e-6:
        return np.eye(3)
    else:
        axis = so3 / theta
        return axis2rot(axis, theta)


def so32quat(so3: np.ndarray) -> np.ndarray:
    assert so3.shape == (3, ), "so32quat so3 shape err"

    axis, theta = so32axis(so3)
    return axis2quat(axis, theta)


def so32axis(so3: np.ndarray) -> Tuple[np.ndarray, float]:
    assert so3.shape == (3, ), "so32axis so3 shape err"
    theta = np.linalg.norm(so3)
    if theta < 1e-6:
        return np.array([1.0, 0.0, 0.0]), 0.0
    else:
        axis = so3 / theta
        return axis, theta


def trans2part(trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert trans.shape == (4, 4), "trans2part trans shape err"

    pos = trans[:3, 3]
    quat = rot2quat(trans[:3, :3])
    return pos, quat


def trans2se3(trans: np.ndarray) -> np.ndarray:
    assert trans.shape == (4, 4), "trans2se3 trans shape err"

    return np.concatenate((trans[:3, 3], rot2so3(trans[:3, :3])))


def part2trans(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    assert pos.shape == (3, ), "part2trans pos shape err"
    assert quat.shape == (4, ), "part2trans quat shape err"

    trans = np.eye(4)
    trans[:3, 3] = pos
    trans[:3, :3] = quat2rot(quat)
    return trans


def part2se3(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    assert pos.shape == (3, ), "part2se3 pos shape err"
    assert quat.shape == (4, ), "part2se3 quat shape err"

    se3 = np.concatenate((pos, quat2so3(quat)))
    return se3


def se32trans(se3: np.ndarray) -> np.ndarray:
    assert se3.shape == (6, ), "se32trans se3 shape err"

    trans = np.eye(4)
    trans[:3, 3] = se3[:3]
    trans[:3, :3] = so32rot(se3[3:])
    return trans


def se32part(se3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert se3.shape == (6, ), "se32part se3 shape err"

    return se3[:3], so32quat(se3[3:])


def zyz2rot(zyz: np.ndarray) -> np.ndarray:
    assert zyz.shape == (3, ), "zyz2rot zyz shape err"

    # temp vars
    theta1, theta2, theta3 = zyz
    cos1 = np.cos(theta1)
    sin1 = np.sin(theta1)
    cos2 = np.cos(theta2)
    sin2 = np.sin(theta2)
    cos3 = np.cos(theta3)
    sin3 = np.sin(theta3)

    # rot
    rot = np.array([
        [
            cos1 * cos2 * cos3 - sin1 * sin3,
            -cos1 * cos2 * sin3 - sin1 * cos3,
            cos1 * sin2,
        ],
        [
            sin1 * cos2 * cos3 + cos1 * sin3,
            -sin1 * cos2 * sin3 + cos1 * cos3,
            sin1 * sin2,
        ],
        [
            -sin2 * cos3,
            sin2 * sin3,
            cos2,
        ],
    ])
    return rot


def rot2zyz(rot: np.ndarray, neg: bool = False) -> np.ndarray:
    assert rot.shape == (3, 3), "rot2zyz rot shape err"

    # positive
    theta1 = np.arctan2(rot[1, 2], rot[0, 2])
    theta2 = np.arccos(rot[2, 2])
    theta3 = np.arctan2(rot[2, 1], -rot[2, 0])

    # negative
    if neg:
        theta1 = angle_norm(theta1 + np.pi)
        theta2 = -theta2
        theta3 = angle_norm(theta3 + np.pi)

    return np.array([theta1, theta2, theta3])


def yaw2quat(yaw: float) -> np.ndarray:
    quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    return quat


def quat2yaw(quat: np.ndarray) -> float:
    assert quat.shape == (4, ), "quat2yaw quat shape err"

    yaw = 2 * np.arctan2(quat[3], quat[0])
    return yaw
