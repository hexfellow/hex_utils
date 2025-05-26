#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-05-25
################################################################

import copy
import numpy as np
import pinocchio as pin
from typing import Tuple

from hex_utils._math_util import part2trans, trans_inv, trans2se3
from hex_utils._hex_arm_state import HexArmState
from hex_utils._hex_cart_pose import HexCartPose


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
        self.__joint_num = self.__model.njoints
        self.__end_link_id = self.__model.getFrameId(end_effector)
        self.__end_joint_id = self.__joint_num - 1

        ### gravity vector
        self.__model.gravity.linear = gravity

    def update_gravity(
            self,
            gravity: np.ndarray = np.array([0, 0, -9.81]),
    ):
        self.__model.gravity.linear = gravity

    # get [M(q), C(q, q_dot), G(q), J(q), J_dot(q, q_dot)]
    # v = J @ q_dot
    def dynamic_params(
        self,
        arm_state: HexArmState,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = arm_state.get_pos()
        dq = arm_state.get_vel()

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
    def forward_kinematics(self, arm_state: HexArmState) -> np.ndarray:
        q = arm_state.get_pos()

        # Compute forward kinematics to update joint placements
        pin.forwardKinematics(self.__model, self.__data, q)

        # Collect the poses of all joints
        joint_poses = np.zeros((self.__joint_num - 1, 4, 4))
        for i in range(1, self.__joint_num):
            joint_placement = self.__data.oMi[i]
            joint_poses[i - 1] = joint_placement.homogeneous

        return joint_poses

    def inverse_kinematics(
        self,
        target_pose: HexCartPose,
        start_joints: HexArmState,
        dt: float = 1e-1,
        exit_eps: float = 1e-3,
        feasible_eps: float = 1e-2,
        damp: float = 1e-12,
        max_iter: int = 300,
    ) -> np.ndarray:
        result_joints = copy.deepcopy(start_joints.get_pos())
        trans_tar_in_base = copy.deepcopy(
            part2trans(
                target_pose.get_pos(),
                target_pose.get_quat(),
            ))
        trans_base_in_tar = trans_inv(trans_tar_in_base)

        result_flag = False
        for _ in range(max_iter):
            pin.forwardKinematics(self.__model, self.__data, result_joints)
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
                result_joints,
                self.__end_joint_id,
            )
            vel = -jac.T @ np.linalg.solve(jac @ jac.T + damp * np.eye(6), err)
            result_joints = pin.integrate(
                self.__model,
                result_joints,
                vel * dt,
            )

        if err_norm < feasible_eps:
            result_flag = True

        return result_flag, result_joints, err_norm
