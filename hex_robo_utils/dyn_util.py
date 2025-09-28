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
from hex_robo_utils.math_utils import angle_norm


class DynUtil:

    def __init__(
            self,
            model_path: str,
            end_effector: str,
            gravity: np.ndarray = np.array([0, 0, -9.81]),
    ):
        ### pinocchio init
        self.__model = pin.buildModelFromUrdf(model_path)
        self.__data = self.__model.createData()
        self.__joint_num = self.__model.njoints - 1
        self.__end_link_id = self.__model.getFrameId(end_effector)
        self.__end_joint_id = self.__joint_num
        self.__lower_limit = self.__model.lowerPositionLimit
        self.__upper_limit = self.__model.upperPositionLimit

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
        c_mat = self.__data.C
        g_vec = self.__data.g
        pin.computeJointJacobians(
            self.__model,
            self.__data,
            q,
        )
        jac = pin.getFrameJacobian(
            self.__model,
            self.__data,
            self.__end_link_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        pin.computeJointJacobiansTimeVariation(
            self.__model,
            self.__data,
            q,
            dq,
        )
        jac_dot = pin.getFrameJacobianTimeVariation(
            self.__model,
            self.__data,
            self.__end_link_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        return m_mat, c_mat, g_vec, jac, jac_dot

    # get [pose_1, pose_2, ..., pose_n]
    def forward_kinematics(
        self,
        q: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:

        # Compute forward kinematics to update joint placements
        pin.forwardKinematics(self.__model, self.__data, q)

        # Collect the poses of all joints
        joint_poses = []
        for i in range(self.__joint_num):
            trans = self.__data.oMi[i + 1].homogeneous
            pos, quat = trans2part(trans)
            joint_poses.append((pos, quat))

        return joint_poses

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
        trans_tar_in_base = copy.deepcopy(
            part2trans(
                tar_pose[0],
                tar_pose[1],
            ))
        trans_base_in_tar = trans_inv(trans_tar_in_base)

        # inverse kinematics
        result_flag = False
        for _ in range(max_iter):
            pin.forwardKinematics(self.__model, self.__data, result_q)
            trans_end_in_base = self.__data.oMi[
                self.__end_joint_id].homogeneous
            trans_tar_in_end = trans_base_in_tar @ trans_end_in_base
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
            vel = -jac.T @ np.linalg.solve(jac @ jac.T + damp * np.eye(6), err)
            result_q = pin.integrate(
                self.__model,
                result_q,
                vel * dt,
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
