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
    assert (vec.ndim == 1 and vec.shape == (3, )) or (
        vec.ndim == 2 and vec.shape[1] == 3), "cross_matrix vec shape err"

    if vec.ndim == 1:
        mat = np.array([
            [0.0, -vec[2], vec[1]],
            [vec[2], 0.0, -vec[0]],
            [-vec[1], vec[0], 0.0],
        ])
    else:
        # Batch processing: input shape (N, 3) -> output shape (N, 3, 3)
        num = vec.shape[0]
        mat = np.zeros((num, 3, 3))
        mat[:, 0, 1] = -vec[:, 2]
        mat[:, 0, 2] = vec[:, 1]
        mat[:, 1, 0] = vec[:, 2]
        mat[:, 1, 2] = -vec[:, 0]
        mat[:, 2, 0] = -vec[:, 1]
        mat[:, 2, 1] = vec[:, 0]
    return mat


def vee(mat: np.ndarray):
    """skew-symmetric matrix → so(3) vector"""
    assert (mat.ndim == 2 and mat.shape == (3, 3)) or (
        mat.ndim == 3 and mat.shape[1:] == (3, 3)), "vee mat shape err"

    if mat.ndim == 2:
        vec = np.array([mat[2, 1], mat[0, 2], mat[1, 0]])
    else:
        # Batch processing: input shape (N, 3, 3) -> output shape (N, 3)
        vec = np.zeros((mat.shape[0], 3))
        vec[:, 0] = mat[:, 2, 1]
        vec[:, 1] = mat[:, 0, 2]
        vec[:, 2] = mat[:, 1, 0]
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


def quat_slerp(q1: np.ndarray, q2: np.ndarray, t) -> np.ndarray:
    assert ((q1.ndim == 1 and q1.shape == (4, ))
            or (q1.ndim == 2 and q1.shape[1] == 4)), "quat_slerp q1 shape err"
    assert ((q2.ndim == 1 and q2.shape == (4, ))
            or (q2.ndim == 2 and q2.shape[1] == 4)), "quat_slerp q2 shape err"
    assert q1.ndim == q2.ndim, "quat_slerp q1 and q2 must have same ndim"
    if q1.ndim == 2:
        assert q1.shape[0] == q2.shape[0], "quat_slerp batch size mismatch"
        assert (isinstance(t, np.ndarray) and t.ndim == 1 and
                t.shape[0] == q1.shape[0]) or isinstance(t, (int, float)), \
            "quat_slerp t shape err"

    if q1.ndim == 1:
        assert isinstance(
            t, (int, float)), "quat_slerp t must be scalar for 1D quat"
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
    else:
        # Batch processing: input shape (N, 4), (N, 4), (N,) or scalar -> output shape (N, 4)
        if isinstance(t, (int, float)):
            t = np.full(q1.shape[0], t)

        # normalize
        q1_norm = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
        neg_mask1 = q1_norm[:, 0] < 0.0
        q1_norm[neg_mask1] = -q1_norm[neg_mask1]
        q2_norm = q2 / np.linalg.norm(q2, axis=1, keepdims=True)
        neg_mask2 = q2_norm[:, 0] < 0.0
        q2_norm[neg_mask2] = -q2_norm[neg_mask2]

        # dot
        dot = np.sum(q1_norm * q2_norm, axis=1)
        neg_dot_mask = dot < 0.0
        q2_norm[neg_dot_mask] = -q2_norm[neg_dot_mask]
        dot[neg_dot_mask] = -dot[neg_dot_mask]
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)

        # slerp
        small_mask = np.fabs(theta) < 1e-6
        q = np.zeros_like(q1)

        if np.any(small_mask):
            q[small_mask] = q1_norm[small_mask] + t[small_mask, np.newaxis] * (
                q2_norm[small_mask] - q1_norm[small_mask])
            q[small_mask] = q[small_mask] / np.linalg.norm(
                q[small_mask], axis=1, keepdims=True)

        if np.any(~small_mask):
            sin_theta = np.sin(theta[~small_mask])
            q1_factor = np.sin(
                (1 - t[~small_mask]) * theta[~small_mask]) / sin_theta
            q2_factor = np.sin(t[~small_mask] * theta[~small_mask]) / sin_theta
            q[~small_mask] = (q1_factor[:, np.newaxis] * q1_norm[~small_mask] +
                              q2_factor[:, np.newaxis] * q2_norm[~small_mask])
        return q


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    assert ((q1.ndim == 1 and q1.shape == (4, ))
            or (q1.ndim == 2 and q1.shape[1] == 4)), "quat_mul q1 shape err"
    assert ((q2.ndim == 1 and q2.shape == (4, ))
            or (q2.ndim == 2 and q2.shape[1] == 4)), "quat_mul q2 shape err"
    assert q1.ndim == q2.ndim, "quat_mul q1 and q2 must have same ndim"
    if q1.ndim == 2:
        assert q1.shape[0] == q2.shape[0], "quat_mul batch size mismatch"

    # normalize
    if q1.ndim == 1:
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
    else:
        # Batch processing
        q1_norm = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
        neg_mask1 = q1_norm[:, 0] < 0.0
        q1_norm[neg_mask1] = -q1_norm[neg_mask1]
        q2_norm = q2 / np.linalg.norm(q2, axis=1, keepdims=True)
        neg_mask2 = q2_norm[:, 0] < 0.0
        q2_norm[neg_mask2] = -q2_norm[neg_mask2]

        # mul
        w1, x1, y1, z1 = q1_norm[:, 0], q1_norm[:, 1], q1_norm[:,
                                                               2], q1_norm[:,
                                                                           3]
        w2, x2, y2, z2 = q2_norm[:, 0], q2_norm[:, 1], q2_norm[:,
                                                               2], q2_norm[:,
                                                                           3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        q = np.stack([w, x, y, z], axis=1)
    return q


def quat_inv(quat: np.ndarray) -> np.ndarray:
    assert (quat.ndim == 1 and quat.shape == (4, )) or (
        quat.ndim == 2 and quat.shape[1] == 4), "quat_inv quat shape err"

    if quat.ndim == 1:
        q = quat / np.linalg.norm(quat)
        if q[0] < 0.0:
            q = -q
        # inv
        inv = np.array([q[0], -q[1], -q[2], -q[3]])
    else:
        # Batch processing: input shape (N, 4) -> output shape (N, 4)
        q = quat / np.linalg.norm(quat, axis=1, keepdims=True)
        neg_mask = q[:, 0] < 0.0
        q[neg_mask] = -q[neg_mask]
        # inv
        inv = np.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], axis=1)
    return inv


def trans_inv(trans: np.ndarray) -> np.ndarray:
    assert (trans.ndim == 2 and trans.shape == (4, 4)) or (
        trans.ndim == 3
        and trans.shape[1:] == (4, 4)), "trans_inv trans shape err"

    if trans.ndim == 2:
        pos = trans[:3, 3]
        rot = trans[:3, :3]

        # inv
        inv = np.eye(4)
        inv[:3, :3] = rot.T
        inv[:3, 3] = -inv[:3, :3] @ pos
        return inv
    else:
        # Batch processing: input shape (N, 4, 4) -> output shape (N, 4, 4)
        pos = trans[:, :3, 3]
        rot = trans[:, :3, :3]

        # inv
        inv = np.zeros_like(trans)
        inv[:, :3, :3] = rot.transpose(0, 2, 1)
        inv[:, :3, 3] = -np.einsum('ijk,ik->ij', inv[:, :3, :3], pos)
        inv[:, 3, :3] = 0.0
        inv[:, 3, 3] = 1.0
        return inv


def rot2quat(rot: np.ndarray) -> np.ndarray:
    assert (rot.ndim == 2 and rot.shape == (3, 3)) or (
        rot.ndim == 3 and rot.shape[1:] == (3, 3)), "rot2quat rot shape err"

    if rot.ndim == 2:
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
    else:
        # Batch processing: input shape (N, 3, 3) -> output shape (N, 4)
        num = rot.shape[0]
        quat = np.zeros((num, 4))
        trace = np.trace(rot, axis1=1, axis2=2)

        # Case 1: trace > 0
        mask1 = trace > 0
        if np.any(mask1):
            temp = 2.0 * np.sqrt(1 + trace[mask1])
            quat[mask1, 0] = 0.25 * temp
            quat[mask1, 1] = (rot[mask1, 2, 1] - rot[mask1, 1, 2]) / temp
            quat[mask1, 2] = (rot[mask1, 0, 2] - rot[mask1, 2, 0]) / temp
            quat[mask1, 3] = (rot[mask1, 1, 0] - rot[mask1, 0, 1]) / temp

        # Case 2: trace <= 0
        mask2 = ~mask1
        if np.any(mask2):
            rot_masked = rot[mask2]
            diag = np.diagonal(rot_masked, axis1=1, axis2=2)
            mask2a = (diag[:, 0] > diag[:, 1]) & (diag[:, 0] > diag[:, 2])
            mask2b = (~mask2a) & (diag[:, 1] > diag[:, 2])
            mask2c = (~mask2a) & (~mask2b)

            # Case 2a: rot[0,0] > rot[1,1] and rot[0,0] > rot[2,2]
            if np.any(mask2a):
                idx2a = np.where(mask2)[0][mask2a]
                temp = 2.0 * np.sqrt(1 + diag[mask2a, 0] - diag[mask2a, 1] -
                                     diag[mask2a, 2])
                quat[idx2a, 0] = (rot_masked[mask2a, 2, 1] -
                                  rot_masked[mask2a, 1, 2]) / temp
                quat[idx2a, 1] = 0.25 * temp
                quat[idx2a, 2] = (rot_masked[mask2a, 1, 0] +
                                  rot_masked[mask2a, 0, 1]) / temp
                quat[idx2a, 3] = (rot_masked[mask2a, 0, 2] +
                                  rot_masked[mask2a, 2, 0]) / temp

            # Case 2b: rot[1,1] > rot[2,2]
            if np.any(mask2b):
                idx2b = np.where(mask2)[0][mask2b]
                temp = 2.0 * np.sqrt(1 + diag[mask2b, 1] - diag[mask2b, 0] -
                                     diag[mask2b, 2])
                quat[idx2b, 0] = (rot_masked[mask2b, 0, 2] -
                                  rot_masked[mask2b, 2, 0]) / temp
                quat[idx2b, 1] = (rot_masked[mask2b, 1, 0] +
                                  rot_masked[mask2b, 0, 1]) / temp
                quat[idx2b, 2] = 0.25 * temp
                quat[idx2b, 3] = (rot_masked[mask2b, 2, 1] +
                                  rot_masked[mask2b, 1, 2]) / temp

            # Case 2c: rot[2,2] >= rot[0,0] and rot[2,2] >= rot[1,1]
            if np.any(mask2c):
                idx2c = np.where(mask2)[0][mask2c]
                temp = 2.0 * np.sqrt(1 + diag[mask2c, 2] - diag[mask2c, 0] -
                                     diag[mask2c, 1])
                quat[idx2c, 0] = (rot_masked[mask2c, 1, 0] -
                                  rot_masked[mask2c, 0, 1]) / temp
                quat[idx2c, 1] = (rot_masked[mask2c, 0, 2] +
                                  rot_masked[mask2c, 2, 0]) / temp
                quat[idx2c, 2] = (rot_masked[mask2c, 2, 1] +
                                  rot_masked[mask2c, 1, 2]) / temp
                quat[idx2c, 3] = 0.25 * temp

        return quat


def rot2axis(rot: np.ndarray) -> Tuple[np.ndarray, float]:
    assert (rot.ndim == 2 and rot.shape == (3, 3)) or (
        rot.ndim == 3 and rot.shape[1:] == (3, 3)), "rot2axis rot shape err"

    if rot.ndim == 2:
        cos_theta = 0.5 * (np.trace(rot) - 1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        if theta < 1e-6:
            return np.array([1.0, 0.0, 0.0]), 0.0
        else:
            axis_matrix = (rot - rot.T) / (2 * np.sin(theta))
            axis = vee(axis_matrix)
        return axis, theta
    else:
        # Batch processing: input shape (N, 3, 3) -> output (N, 3), (N,)
        cos_theta = 0.5 * (np.trace(rot, axis1=1, axis2=2) - 1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        small_mask = theta < 1e-6
        axis = np.zeros((rot.shape[0], 3))
        axis[small_mask] = np.array([1.0, 0.0, 0.0])

        if np.any(~small_mask):
            axis_matrix = (rot[~small_mask] -
                           rot[~small_mask].transpose(0, 2, 1)) / (2 * np.sin(
                               theta[~small_mask, np.newaxis, np.newaxis]))
            axis[~small_mask] = vee(axis_matrix)

        return axis, theta


def rot2so3(rot: np.ndarray) -> np.ndarray:
    assert (rot.ndim == 2 and rot.shape == (3, 3)) or (
        rot.ndim == 3 and rot.shape[1:] == (3, 3)), "rot2so3 rot shape err"

    axis, theta = rot2axis(rot)
    if rot.ndim == 2:
        return theta * axis
    else:
        # Batch processing: theta is (N,), axis is (N, 3)
        return theta[:, np.newaxis] * axis


def quat2rot(quat: np.ndarray) -> np.ndarray:
    assert (quat.ndim == 1 and quat.shape == (4, )) or (
        quat.ndim == 2 and quat.shape[1] == 4), "quat2rot quat shape err"

    if quat.ndim == 1:
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
    else:
        # Batch processing: input shape (N, 4) -> output shape (N, 3, 3)
        q = quat / np.linalg.norm(quat, axis=1, keepdims=True)
        neg_mask = q[:, 0] < 0.0
        q[neg_mask] = -q[neg_mask]

        # temp vars
        qx2 = q[:, 1] * q[:, 1]
        qy2 = q[:, 2] * q[:, 2]
        qz2 = q[:, 3] * q[:, 3]
        qxqw = q[:, 1] * q[:, 0]
        qyqw = q[:, 2] * q[:, 0]
        qzqw = q[:, 3] * q[:, 0]
        qxqy = q[:, 1] * q[:, 2]
        qyqz = q[:, 2] * q[:, 3]
        qzqx = q[:, 3] * q[:, 1]

        # rot
        num = q.shape[0]
        rot = np.zeros((num, 3, 3))
        rot[:, 0, 0] = 1 - 2 * (qy2 + qz2)
        rot[:, 0, 1] = 2 * (qxqy - qzqw)
        rot[:, 0, 2] = 2 * (qzqx + qyqw)
        rot[:, 1, 0] = 2 * (qxqy + qzqw)
        rot[:, 1, 1] = 1 - 2 * (qx2 + qz2)
        rot[:, 1, 2] = 2 * (qyqz - qxqw)
        rot[:, 2, 0] = 2 * (qzqx - qyqw)
        rot[:, 2, 1] = 2 * (qyqz + qxqw)
        rot[:, 2, 2] = 1 - 2 * (qx2 + qy2)
    return rot


def quat2axis(quat: np.ndarray) -> Tuple[np.ndarray, float]:
    assert (quat.ndim == 1 and quat.shape == (4, )) or (
        quat.ndim == 2 and quat.shape[1] == 4), "quat2axis quat shape err"

    if quat.ndim == 1:
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
    else:
        # Batch processing: input shape (N, 4) -> output (N, 3), (N,)
        q = quat / np.linalg.norm(quat, axis=1, keepdims=True)
        neg_mask = q[:, 0] < 0.0
        q[neg_mask] = -q[neg_mask]

        vec = q[:, 1:]
        norm_vec = np.linalg.norm(vec, axis=1)
        small_mask = norm_vec < 1e-6

        axis = np.zeros((q.shape[0], 3))
        axis[~small_mask] = vec[~small_mask] / norm_vec[~small_mask,
                                                        np.newaxis]
        axis[small_mask] = np.array([1.0, 0.0, 0.0])

        theta = np.zeros(q.shape[0])
        theta[~small_mask] = 2 * np.arctan2(norm_vec[~small_mask],
                                            q[~small_mask, 0])
        return axis, theta


def quat2so3(quat: np.ndarray) -> np.ndarray:
    assert (quat.ndim == 1 and quat.shape == (4, )) or (
        quat.ndim == 2 and quat.shape[1] == 4), "quat2so3 quat shape err"

    axis, theta = quat2axis(quat)
    if quat.ndim == 1:
        return theta * axis
    else:
        # Batch processing: theta is (N,), axis is (N, 3)
        return theta[:, np.newaxis] * axis


def axis2rot(axis: np.ndarray, theta) -> np.ndarray:
    assert (axis.ndim == 1 and axis.shape == (3, )) or (
        axis.ndim == 2 and axis.shape[1] == 3), "axis2rot axis shape err"
    if axis.ndim == 1:
        assert isinstance(
            theta, (int, float)), "axis2rot theta must be scalar for 1D axis"
    else:
        assert (isinstance(theta, np.ndarray) and theta.ndim == 1 and
                theta.shape[0] == axis.shape[0]) or isinstance(theta, (int, float)), \
            "axis2rot theta shape err"

    if axis.ndim == 1:
        if theta < 1e-6:
            return np.eye(3)

        axis_matrix = hat(axis)
        rot = np.eye(3) + np.sin(theta) * axis_matrix + (1 - np.cos(theta)) * (
            axis_matrix @ axis_matrix)
        return rot
    else:
        # Batch processing: input shape (N, 3), (N,) or scalar -> output shape (N, 3, 3)
        if isinstance(theta, (int, float)):
            theta = np.full(axis.shape[0], theta)

        num = axis.shape[0]
        rot = np.zeros((num, 3, 3))
        small_mask = theta < 1e-6
        rot[small_mask] = np.eye(3)

        if np.any(~small_mask):
            axis_matrix = hat(axis[~small_mask])
            sin_theta = np.sin(theta[~small_mask])
            cos_theta = np.cos(theta[~small_mask])
            # Batch matrix multiplication
            axis_matrix_sq = axis_matrix @ axis_matrix
            rot[~small_mask] = (
                np.eye(3)[np.newaxis, :, :] +
                sin_theta[:, np.newaxis, np.newaxis] * axis_matrix +
                (1 - cos_theta)[:, np.newaxis, np.newaxis] * axis_matrix_sq)
        return rot


def axis2quat(axis: np.ndarray, theta) -> np.ndarray:
    assert (axis.ndim == 1 and axis.shape == (3, )) or (
        axis.ndim == 2 and axis.shape[1] == 3), "axis2quat axis shape err"
    if axis.ndim == 1:
        assert isinstance(
            theta, (int, float)), "axis2quat theta must be scalar for 1D axis"
    else:
        assert (isinstance(theta, np.ndarray) and theta.ndim == 1 and
                theta.shape[0] == axis.shape[0]) or isinstance(theta, (int, float)), \
            "axis2quat theta shape err"

    if axis.ndim == 1:
        if theta < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0])

        quat = np.zeros(4)
        quat[0] = np.cos(theta / 2)
        quat[1:] = axis * np.sin(theta / 2)
        return quat
    else:
        # Batch processing: input shape (N, 3), (N,) or scalar -> output shape (N, 4)
        if isinstance(theta, (int, float)):
            theta = np.full(axis.shape[0], theta)

        num = axis.shape[0]
        quat = np.zeros((num, 4))
        small_mask = theta < 1e-6
        quat[small_mask] = np.array([1.0, 0.0, 0.0, 0.0])

        if np.any(~small_mask):
            half_theta = theta[~small_mask] / 2
            quat[~small_mask, 0] = np.cos(half_theta)
            quat[~small_mask,
                 1:] = axis[~small_mask] * np.sin(half_theta)[:, np.newaxis]
        return quat


def axis2so3(axis: np.ndarray, theta) -> np.ndarray:
    assert (axis.ndim == 1 and axis.shape == (3, )) or (
        axis.ndim == 2 and axis.shape[1] == 3), "axis2so3 axis shape err"
    if axis.ndim == 1:
        assert isinstance(
            theta, (int, float)), "axis2so3 theta must be scalar for 1D axis"
    else:
        assert (isinstance(theta, np.ndarray) and theta.ndim == 1 and
                theta.shape[0] == axis.shape[0]) or isinstance(theta, (int, float)), \
            "axis2so3 theta shape err"

    if axis.ndim == 1:
        return theta * axis
    else:
        # Batch processing: input shape (N, 3), (N,) or scalar -> output shape (N, 3)
        if isinstance(theta, (int, float)):
            theta = np.full(axis.shape[0], theta)
        return theta[:, np.newaxis] * axis


def so32rot(so3: np.ndarray) -> np.ndarray:
    assert (so3.ndim == 1 and so3.shape == (3, )) or (
        so3.ndim == 2 and so3.shape[1] == 3), "so32rot so3 shape err"

    if so3.ndim == 1:
        theta = np.linalg.norm(so3)
        if theta < 1e-6:
            return np.eye(3)
        else:
            axis = so3 / theta
            return axis2rot(axis, theta)
    else:
        # Batch processing: input shape (N, 3) -> output shape (N, 3, 3)
        theta = np.linalg.norm(so3, axis=1)
        axis = so3 / theta[:, np.newaxis]
        return axis2rot(axis, theta)


def so32quat(so3: np.ndarray) -> np.ndarray:
    assert (so3.ndim == 1 and so3.shape == (3, )) or (
        so3.ndim == 2 and so3.shape[1] == 3), "so32quat so3 shape err"

    axis, theta = so32axis(so3)
    return axis2quat(axis, theta)


def so32axis(so3: np.ndarray) -> Tuple[np.ndarray, float]:
    assert (so3.ndim == 1 and so3.shape == (3, )) or (
        so3.ndim == 2 and so3.shape[1] == 3), "so32axis so3 shape err"

    if so3.ndim == 1:
        theta = np.linalg.norm(so3)
        if theta < 1e-6:
            return np.array([1.0, 0.0, 0.0]), 0.0
        else:
            axis = so3 / theta
            return axis, theta
    else:
        # Batch processing: input shape (N, 3) -> output (N, 3), (N,)
        theta = np.linalg.norm(so3, axis=1)
        small_mask = theta < 1e-6
        axis = np.zeros((so3.shape[0], 3))
        axis[small_mask] = np.array([1.0, 0.0, 0.0])
        axis[~small_mask] = so3[~small_mask] / theta[~small_mask, np.newaxis]
        return axis, theta


def trans2part(trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert (trans.ndim == 2 and trans.shape == (4, 4)) or (
        trans.ndim == 3
        and trans.shape[1:] == (4, 4)), "trans2part trans shape err"

    if trans.ndim == 2:
        pos = trans[:3, 3]
        quat = rot2quat(trans[:3, :3])
        return pos, quat
    else:
        # Batch processing: input shape (N, 4, 4) -> output (N, 3), (N, 4)
        pos = trans[:, :3, 3]
        quat = rot2quat(trans[:, :3, :3])
        return pos, quat


def trans2se3(trans: np.ndarray) -> np.ndarray:
    assert (trans.ndim == 2 and trans.shape == (4, 4)) or (
        trans.ndim == 3
        and trans.shape[1:] == (4, 4)), "trans2se3 trans shape err"

    if trans.ndim == 2:
        return np.concatenate((trans[:3, 3], rot2so3(trans[:3, :3])))
    else:
        # Batch processing: input shape (N, 4, 4) -> output shape (N, 6)
        pos = trans[:, :3, 3]
        so3 = rot2so3(trans[:, :3, :3])
        return np.concatenate((pos, so3), axis=1)


def part2trans(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    assert (pos.ndim == 1 and pos.shape == (3, )) or (
        pos.ndim == 2 and pos.shape[1] == 3), "part2trans pos shape err"
    assert (quat.ndim == 1 and quat.shape == (4, )) or (
        quat.ndim == 2 and quat.shape[1] == 4), "part2trans quat shape err"
    assert pos.ndim == quat.ndim, "part2trans pos and quat must have same ndim"
    if pos.ndim == 2:
        assert pos.shape[0] == quat.shape[0], "part2trans batch size mismatch"

    if pos.ndim == 1:
        trans = np.eye(4)
        trans[:3, 3] = pos
        trans[:3, :3] = quat2rot(quat)
        return trans
    else:
        # Batch processing: input shape (N, 3), (N, 4) -> output shape (N, 4, 4)
        num = pos.shape[0]
        trans = np.zeros((num, 4, 4))
        trans[:, :3, 3] = pos
        trans[:, :3, :3] = quat2rot(quat)
        trans[:, 3, :3] = 0.0
        trans[:, 3, 3] = 1.0
        return trans


def part2se3(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    assert (pos.ndim == 1 and pos.shape == (3, )) or (
        pos.ndim == 2 and pos.shape[1] == 3), "part2se3 pos shape err"
    assert (quat.ndim == 1 and quat.shape == (4, )) or (
        quat.ndim == 2 and quat.shape[1] == 4), "part2se3 quat shape err"
    assert pos.ndim == quat.ndim, "part2se3 pos and quat must have same ndim"
    if pos.ndim == 2:
        assert pos.shape[0] == quat.shape[0], "part2se3 batch size mismatch"

    if pos.ndim == 1:
        se3 = np.concatenate((pos, quat2so3(quat)))
        return se3
    else:
        # Batch processing: input shape (N, 3), (N, 4) -> output shape (N, 6)
        so3 = quat2so3(quat)
        return np.concatenate((pos, so3), axis=1)


def se32trans(se3: np.ndarray) -> np.ndarray:
    assert (se3.ndim == 1 and se3.shape == (6, )) or (
        se3.ndim == 2 and se3.shape[1] == 6), "se32trans se3 shape err"

    if se3.ndim == 1:
        trans = np.eye(4)
        trans[:3, 3] = se3[:3]
        trans[:3, :3] = so32rot(se3[3:])
        return trans
    else:
        # Batch processing: input shape (N, 6) -> output shape (N, 4, 4)
        num = se3.shape[0]
        trans = np.zeros((num, 4, 4))
        trans[:, :3, 3] = se3[:, :3]
        trans[:, :3, :3] = so32rot(se3[:, 3:])
        trans[:, 3, :3] = 0.0
        trans[:, 3, 3] = 1.0
        return trans


def se32part(se3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert (se3.ndim == 1 and se3.shape == (6, )) or (
        se3.ndim == 2 and se3.shape[1] == 6), "se32part se3 shape err"

    if se3.ndim == 1:
        return se3[:3], so32quat(se3[3:])
    else:
        # Batch processing: input shape (N, 6) -> output (N, 3), (N, 4)
        return se3[:, :3], so32quat(se3[:, 3:])


def zyz2rot(zyz: np.ndarray) -> np.ndarray:
    assert (zyz.ndim == 1 and zyz.shape == (3, )) or (
        zyz.ndim == 2 and zyz.shape[1] == 3), "zyz2rot zyz shape err"

    if zyz.ndim == 1:
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
    else:
        # Batch processing: input shape (N, 3) -> output shape (N, 3, 3)
        theta1, theta2, theta3 = zyz[:, 0], zyz[:, 1], zyz[:, 2]
        cos1 = np.cos(theta1)
        sin1 = np.sin(theta1)
        cos2 = np.cos(theta2)
        sin2 = np.sin(theta2)
        cos3 = np.cos(theta3)
        sin3 = np.sin(theta3)

        # rot
        num = zyz.shape[0]
        rot = np.zeros((num, 3, 3))
        rot[:, 0, 0] = cos1 * cos2 * cos3 - sin1 * sin3
        rot[:, 0, 1] = -cos1 * cos2 * sin3 - sin1 * cos3
        rot[:, 0, 2] = cos1 * sin2
        rot[:, 1, 0] = sin1 * cos2 * cos3 + cos1 * sin3
        rot[:, 1, 1] = -sin1 * cos2 * sin3 + cos1 * cos3
        rot[:, 1, 2] = sin1 * sin2
        rot[:, 2, 0] = -sin2 * cos3
        rot[:, 2, 1] = sin2 * sin3
        rot[:, 2, 2] = cos2
        return rot


def rot2zyz(rot: np.ndarray, neg: bool = False) -> np.ndarray:
    assert (rot.ndim == 2 and rot.shape == (3, 3)) or (
        rot.ndim == 3 and rot.shape[1:] == (3, 3)), "rot2zyz rot shape err"

    if rot.ndim == 2:
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
    else:
        # Batch processing: input shape (N, 3, 3) -> output shape (N, 3)
        # positive
        theta1 = np.arctan2(rot[:, 1, 2], rot[:, 0, 2])
        theta2 = np.arccos(rot[:, 2, 2])
        theta3 = np.arctan2(rot[:, 2, 1], -rot[:, 2, 0])

        # negative
        if neg:
            theta1 = angle_norm(theta1 + np.pi)
            theta2 = -theta2
            theta3 = angle_norm(theta3 + np.pi)

        return np.stack([theta1, theta2, theta3], axis=1)


def yaw2quat(yaw) -> np.ndarray:
    if isinstance(yaw, (int, float)):
        quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        return quat
    else:
        # Batch processing: input shape (N,) -> output shape (N, 4)
        assert isinstance(
            yaw, np.ndarray) and yaw.ndim == 1, "yaw2quat yaw shape err"
        quat = np.zeros((yaw.shape[0], 4))
        quat[:, 0] = np.cos(yaw / 2)
        quat[:, 3] = np.sin(yaw / 2)
        return quat


def quat2yaw(quat: np.ndarray):
    assert (quat.ndim == 1 and quat.shape == (4, )) or (
        quat.ndim == 2 and quat.shape[1] == 4), "quat2yaw quat shape err"

    if quat.ndim == 1:
        yaw = 2 * np.arctan2(quat[3], quat[0])
        return yaw
    else:
        # Batch processing: input shape (N, 4) -> output shape (N,)
        yaw = 2 * np.arctan2(quat[:, 3], quat[:, 0])
        return yaw
