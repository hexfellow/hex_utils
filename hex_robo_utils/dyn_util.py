#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-09-18
################################################################

import copy
import numpy as np
import pinocchio as pin
from typing import Tuple, List

from hex_robo_utils.math_utils import trans2part, part2trans
from hex_robo_utils.math_utils import trans_inv, trans2se3
from hex_robo_utils.math_utils import angle_norm, hat


class HexDynUtil:

    def __init__(
            self,
            model_path: str,
            last_link: str,
            end_pose: np.ndarray = np.array(
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            gravity: np.ndarray = np.array([0, 0, -9.81]),
    ):
        ### pinocchio init
        self.__model = pin.buildModelFromUrdf(model_path)
        self.__data = self.__model.createData()
        self.__joint_num = self.__model.njoints - 1
        self.__end_link_id = self.__model.getFrameId(last_link)
        self.__end_joint_id = self.__joint_num
        self.__lower_limit = self.__model.lowerPositionLimit
        self.__upper_limit = self.__model.upperPositionLimit
        self.__trans_end_in_last = part2trans(end_pose[:3], end_pose[3:])
        self.__trans_last_in_end = trans_inv(self.__trans_end_in_last)
        self.__jac_trans = self.__cal_jac_trans(self.__trans_end_in_last)

        ### gravity vector
        self.__model.gravity.linear = gravity

    def get_gravity(self) -> np.ndarray:
        return copy.deepcopy(self.__model.gravity.linear)

    def set_gravity(
            self,
            gravity: np.ndarray = np.array([0, 0, -9.81]),
    ):
        self.__model.gravity.linear = copy.deepcopy(gravity)

    def get_joint_num(self) -> int:
        return self.__joint_num

    def __cal_jac_trans(self, trans_end_in_last: np.ndarray) -> np.ndarray:
        rot = trans_end_in_last[:3, :3]
        pos = trans_end_in_last[:3, 3]

        jac_trans = np.eye(6)
        jac_trans[:3, :3] = rot.T
        jac_trans[3:, :3] = -rot.T @ hat(pos)
        jac_trans[3:, 3:] = rot.T
        return jac_trans

    # get [M(q), C(q, q_dot), G(q), J(q), J_dot(q, q_dot)]
    # v = J @ q_dot
    def dynamic_params(
        self,
        q: np.ndarray,
        dq: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Compute all dynamic parameters
        pin.computeAllTerms(self.__model, self.__data, q, dq)
        m_mat = self.__data.M
        c_mat = pin.computeCoriolisMatrix(self.__model, self.__data, q, dq)
        g_vec = self.__data.g
        jac = pin.computeJointJacobian(
            self.__model,
            self.__data,
            q,
            self.__end_joint_id,
        )
        jac_dot = pin.computeJointJacobiansTimeVariation(
            self.__model,
            self.__data,
            q,
            dq,
        )

        jac = self.__jac_trans @ jac
        jac_dot = self.__jac_trans @ jac_dot
        return m_mat, c_mat, g_vec, jac, jac_dot

    # get [pose_1, pose_2, ..., pose_n]
    def forward_kinematics(
        self,
        q: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:

        # Compute forward kinematics to update joint placements
        pin.forwardKinematics(self.__model, self.__data, q)

        # Collect the poses of all joints
        poses = []
        trans = None
        for i in range(self.__joint_num):
            trans = self.__data.oMi[i + 1].homogeneous
            pos, quat = trans2part(trans)
            poses.append((pos, quat))
        trans_end_in_base = trans @ self.__trans_end_in_last
        pos, quat = trans2part(trans_end_in_base)
        poses.append((pos, quat))
        return poses

    def inverse_kinematics(
        self,
        tar_pose: Tuple[np.ndarray, np.ndarray],
        start_q: np.ndarray,
        dt: float = 1e-1,
        exit_eps: float = 1e-3,
        feasible_eps: float = 1e-2,
        damp: float = 1e-12,
        max_iter: int = 300,
    ) -> Tuple[bool, np.ndarray, float]:
        result_q = copy.deepcopy(start_q)
        trans_end_tar_in_base = copy.deepcopy(
            part2trans(
                tar_pose[0],
                tar_pose[1],
            ))
        trans_tar_in_base = trans_end_tar_in_base @ self.__trans_last_in_end
        trans_base_in_tar = trans_inv(trans_tar_in_base)

        # inverse kinematics
        result_flag = False
        for _ in range(max_iter):
            pin.forwardKinematics(self.__model, self.__data, result_q)
            trans_end_in_base = self.__data.oMi[
                self.__end_joint_id].homogeneous
            trans_tar_in_end = trans_inv(trans_end_in_base) @ trans_tar_in_base
            err = trans2se3(trans_tar_in_end)

            err_norm = np.linalg.norm(err)
            if err_norm < exit_eps:
                result_flag = True
                break

            # jac in joint frame
            jac = pin.computeJointJacobian(
                self.__model,
                self.__data,
                result_q,
                self.__end_joint_id,
            )
            dq = np.linalg.pinv(jac, rcond=damp) @ err
            result_q = pin.integrate(
                self.__model,
                result_q,
                dq * dt,
            )

        # post process
        result_q = np.clip(
            angle_norm(result_q),
            self.__lower_limit,
            self.__upper_limit,
        )

        # check feasible
        pin.forwardKinematics(self.__model, self.__data, result_q)
        trans_end_in_base = self.__data.oMi[self.__end_joint_id].homogeneous
        trans_tar_in_end = trans_base_in_tar @ trans_end_in_base
        err = trans2se3(trans_tar_in_end)
        err_norm = np.linalg.norm(err)
        if err_norm < feasible_eps:
            result_flag = True

        return result_flag, result_q, err_norm


class HexFricUtil:
    # tanh-based friction model
    # tau_f = fc * tanh(k * dq) + fv * dq + fo
    def __init__(
            self,
            fc: np.ndarray = np.array([1.0] * 6),
            k: np.ndarray = np.array([100.0] * 6),
            fv: np.ndarray = np.array([1.0] * 6),
            fo: np.ndarray = np.array([0.0] * 6),
    ):
        # constants
        self.__fc = fc.copy()
        self.__k = k.copy()
        self.__fv = fv.copy()
        self.__fo = fo.copy()

    def __call__(self, dq: np.ndarray):
        tau_c = self.__fc * np.tanh(self.__k * dq)
        tau_v = self.__fv * dq
        tau_o = self.__fo
        tau_f = tau_c + tau_v + tau_o
        return tau_f
