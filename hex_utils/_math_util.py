#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-01-14
################################################################

import numpy as np
from typing import Tuple


def cross_matrix(vec: np.ndarray):
    assert vec.shape == (3, ), "cross_matrix vec shape err"

    trans = np.array([
        [0.0, -vec[2], vec[1]],
        [vec[2], 0.0, -vec[0]],
        [-vec[1], vec[0], 0.0],
    ])
    return trans


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
    q2_norm = q2 / np.linalg.norm(q2)

    # dot
    dot = np.dot(q1_norm, q2_norm)
    if dot < 0.0:
        q2_norm = -q2_norm
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)

    # slerp
    if np.fabs(theta) < 1e-6:
        q = q1 + t * (q2 - q1)
        q = q / np.linalg.norm(q)
        return q

    sin_theta = np.sin(theta)
    q1_factor = np.sin((1 - t) * theta) / sin_theta
    q2_factor = np.sin(t * theta) / sin_theta
    q = q1_factor * q1 + q2_factor * q2
    return q


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    assert q1.shape == (4, ) and q2.shape == (4, ), "quat_mul quat shape err"

    # normalize
    q1_norm = q1 / np.linalg.norm(q1)
    q2_norm = q2 / np.linalg.norm(q2)

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

    # inv
    inv = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
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


def quat2rot(quat: np.ndarray) -> np.ndarray:
    assert quat.shape == (4, ), "quat2rot quat shape err"

    # normalize
    q = quat / np.linalg.norm(quat)

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


def rot2quat(rot: np.ndarray) -> np.ndarray:
    assert rot.shape == (3, 3), "rot2quat rot shape err"

    qw, qx, qy, qz = 1, 0, 0, 0
    if 1 + rot[0, 0] + rot[1, 1] + rot[2, 2] > 0:
        qw = np.sqrt(1 + rot[0, 0] + rot[1, 1] + rot[2, 2]) / 2
        qx = (rot[2, 1] - rot[1, 2]) / (4 * qw)
        qy = (rot[0, 2] - rot[2, 0]) / (4 * qw)
        qz = (rot[1, 0] - rot[0, 1]) / (4 * qw)
    elif 1 + rot[0, 0] - rot[1, 1] - rot[2, 2] > 0:
        qx = np.sqrt(1 + rot[0, 0] - rot[1, 1] - rot[2, 2]) / 2
        qw = (rot[2, 1] - rot[1, 2]) / (4 * qx)
        qy = (rot[1, 0] + rot[0, 1]) / (4 * qx)
        qz = (rot[0, 2] + rot[2, 0]) / (4 * qx)
    elif 1 - rot[0, 0] + rot[1, 1] - rot[2, 2] > 0:
        qy = np.sqrt(1 - rot[0, 0] + rot[1, 1] - rot[2, 2]) / 2
        qw = (rot[0, 2] - rot[2, 0]) / (4 * qy)
        qx = (rot[1, 0] + rot[0, 1]) / (4 * qy)
        qz = (rot[2, 1] + rot[1, 2]) / (4 * qy)
    elif 1 - rot[0, 0] - rot[1, 1] + rot[2, 2] > 0:
        qz = np.sqrt(1 - rot[0, 0] - rot[1, 1] + rot[2, 2]) / 2
        qw = (rot[1, 0] - rot[0, 1]) / (4 * qz)
        qx = (rot[0, 2] + rot[2, 0]) / (4 * qz)
        qy = (rot[2, 1] + rot[1, 2]) / (4 * qz)
    else:
        raise ValueError("The rotation matrix is not valid!")

    return np.array([qw, qx, qy, qz])


def axis2rot(axis: np.ndarray, angle: float) -> np.ndarray:
    assert axis.shape == (3, ), "axis2rot axis shape err"

    axis_matrix = cross_matrix(axis)
    sin_angle = np.sin(angle)
    cos_angle = np.cos(angle)
    rot = np.eye(3) * cos_angle + (1 - cos_angle) * np.outer(
        axis, axis) + sin_angle * axis_matrix
    return rot


def rot2axis(rot: np.ndarray) -> Tuple[np.ndarray, float]:
    assert rot.shape == (3, 3), "rot2axis rot shape err"
    assert np.trace(rot) <= 3.0 and np.trace(
        rot) >= -1.0, "rot2axis rot track err"

    angle = np.arccos(0.5 * (np.trace(rot) - 1))
    if angle < 1e-6:
        return np.array([1.0, 0.0, 0.0]), 0.0
    else:
        axis = np.array([
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ]) / (2.0 * np.sin(angle))
    return axis, angle


def quat2axis(quat: np.ndarray) -> Tuple[np.ndarray, float]:
    assert quat.shape == (4, ), "quat2axis quat shape err"

    # normalize
    q = quat / np.linalg.norm(quat)

    # angle
    angle = 2 * np.arccos(q[0])

    # axis
    if angle < 1e-6:
        return np.array([1.0, 0.0, 0.0]), 0.0
    else:
        axis = q[1:] / np.sin(angle / 2)
    return axis, angle


def axis2quat(axis: np.ndarray, angle: float) -> np.ndarray:
    assert axis.shape == (3, ), "axis2quat axis shape err"

    quat = np.zeros(4)
    quat[0] = np.cos(angle / 2)
    quat[1:] = axis * np.sin(angle / 2)
    return quat


def part2trans(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    assert pos.shape == (3, ), "part2trans pos shape err"
    assert quat.shape == (4, ), "part2trans quat shape err"

    trans = np.eye(4)
    trans[:3, 3] = pos
    trans[:3, :3] = quat2rot(quat)
    return trans


def trans2part(trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert trans.shape == (4, 4), "trans2part trans shape err"

    pos = trans[:3, 3]
    quat = rot2quat(trans[:3, :3])
    return pos, quat


def se32trans(se3: np.ndarray) -> np.ndarray:
    assert se3.shape == (6, ), "se32trans se3 shape err"

    # temp vars
    pos = se3[:3]
    angle = np.linalg.norm(se3[3:])

    # trans
    if angle < 1e-6:
        return part2trans(pos, np.array([1.0, 0.0, 0.0, 0.0]))
    else:
        axis = se3[3:] / angle
        quat = axis2quat(axis, angle)
        return part2trans(pos, quat)


def trans2se3(trans: np.ndarray) -> np.ndarray:
    assert trans.shape == (4, 4), "trans2se3 trans shape err"

    # temp vars
    pos, quat = trans2part(trans)

    # se3
    angle = 2 * np.arccos(quat[0])
    if angle < 1e-6:
        return np.concatenate((pos, np.zeros(3)))
    else:
        axis = quat[1:] / np.sin(angle / 2)
        return np.concatenate((pos, axis * angle))


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
