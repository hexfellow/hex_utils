#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-05-29
################################################################

import copy
import numpy as np

from hex_utils._hex_arm_state import HexArmState
from hex_utils._hex_cart_pose import HexCartPose
from hex_utils._hex_cart_vel import HexCartVel
from hex_utils._hex_cart_state import HexCartState

from hex_utils._math_utils import quat_slerp
from hex_utils._math_utils import se32trans, trans2part


class ObsUtilJoint:

    def __init__(
        self,
        mass: np.ndarray,
        damp: np.ndarray,
        stiff: np.ndarray,
        dt: float,
        q_limit: np.ndarray,
        dq_limit: np.ndarray,
        ddq_limit: np.ndarray,
    ):
        ### physical params
        self.__mass_inv = np.linalg.inv(mass)
        self.__damp = damp
        self.__stiff = stiff
        self.__dt = dt

        ### limits
        self.__q_limit = q_limit
        self.__dq_limit = dq_limit
        self.__ddq_limit = ddq_limit

        ### variables
        self.__ready = False
        self.__obs_state = None

    def get_mass(self) -> np.ndarray:
        return np.linalg.inv(self.__mass_inv)

    def set_mass(self, mass: np.ndarray):
        self.__mass_inv = np.linalg.inv(mass)

    def get_damp(self) -> np.ndarray:
        return copy.deepcopy(self.__damp)

    def set_damp(self, damp: np.ndarray):
        self.__damp = copy.deepcopy(damp)

    def get_stiff(self) -> np.ndarray:
        return copy.deepcopy(self.__stiff)

    def set_stiff(self, stiff: np.ndarray):
        self.__stiff = copy.deepcopy(stiff)

    def get_dt(self) -> float:
        return self.__dt

    def set_dt(self, dt: float):
        self.__dt = dt

    def is_ready(self) -> bool:
        return self.__ready

    def set_state(self, state: HexArmState):
        self.__obs_state = state
        self.__ready = True

    def get_state(self) -> HexArmState:
        return self.__obs_state

    def predict(
        self,
        state_tar: HexArmState,
    ):
        q_tar = state_tar.get_pos()
        q_cur = self.__obs_state.get_pos()
        dq_cur = self.__obs_state.get_vel()

        # runge-kutta k1
        dq1 = dq_cur
        ddq1 = self.__ddq(q_tar - q_cur, dq_cur)

        # runge-kutta k2
        q2 = q_cur + dq1 * self.__dt * 0.5
        dq2 = dq_cur + ddq1 * self.__dt * 0.5
        ddq2 = self.__ddq(q_tar - q2, dq2)

        # runge-kutta k3
        q3 = q_cur + dq2 * self.__dt * 0.5
        dq3 = dq_cur + ddq2 * self.__dt * 0.5
        ddq3 = self.__ddq(q_tar - q3, dq3)

        # runge-kutta k4
        q4 = q_cur + dq3 * self.__dt
        dq4 = dq_cur + ddq3 * self.__dt
        ddq4 = self.__ddq(q_tar - q4, dq4)

        # runge-kutta
        q_next = q_cur + (dq1 + 2.0 * dq2 + 2.0 * dq3 + dq4) / 6.0 * self.__dt
        dq_next = dq_cur + (ddq1 + 2.0 * ddq2 + 2.0 * ddq3 +
                            ddq4) / 6.0 * self.__dt
        low_mask = q_next < self.__q_limit[:, 0]
        high_mask = q_next > self.__q_limit[:, 1]

        # clip
        q_next[low_mask] = self.__q_limit[low_mask, 0]
        q_next[high_mask] = self.__q_limit[high_mask, 1]
        dq_next[low_mask] = 0.0
        dq_next[high_mask] = 0.0
        dq_next = np.clip(
            dq_next,
            self.__dq_limit[:, 0],
            self.__dq_limit[:, 1],
        )

        # set state
        self.__obs_state.set_pos(q_next)
        self.__obs_state.set_vel(dq_next)

    def __ddq(self, q_err: np.ndarray, dq_cur: np.ndarray) -> np.ndarray:
        ddq = (self.__stiff @ q_err - self.__damp @ dq_cur) @ self.__mass_inv
        ddq = np.clip(ddq, self.__ddq_limit[:, 0], self.__ddq_limit[:, 1])
        return ddq

    def update(self, state_sensor: HexArmState, update_weight: np.ndarray):
        q_sensor = np.clip(state_sensor.get_pos(), self.__q_limit[:, 0],
                           self.__q_limit[:, 1])
        dq_sensor = np.clip(state_sensor.get_vel(), self.__dq_limit[:, 0],
                            self.__dq_limit[:, 1])

        # update state
        obs_weight = 1.0 - update_weight
        self.__obs_state.set_pos(self.__obs_state.get_pos() * obs_weight +
                                 q_sensor * update_weight)
        self.__obs_state.set_vel(self.__obs_state.get_vel() * obs_weight +
                                 dq_sensor * update_weight)


class ObsUtilWork:

    def __init__(
        self,
        dt: float,
        vel_limit: np.ndarray = np.array([
            1.0,  # vel_norm
            1.0,  # omega_norm
        ]),
        acc_limit: np.ndarray = np.array([
            1.0,  # acc_norm
            1.0,  # alpha_norm
        ]),
    ):
        ### physical params
        self.__dt = dt

        ### limits
        self.__vel_lin_limit = vel_limit[0]
        self.__vel_ang_limit = vel_limit[1]
        self.__acc_lin_limit = acc_limit[0]
        self.__acc_ang_limit = acc_limit[1]

        ### variables
        self.__ready = False
        self.__obs_state = None

    def is_ready(self) -> bool:
        return self.__ready

    def set_state(self, state: HexCartState):
        self.__obs_state = state
        self.__ready = True

    def get_state(self) -> HexCartState:
        return self.__obs_state

    def __norm_limit(self, vec: np.ndarray, limit: float) -> np.ndarray:
        vec_norm = np.linalg.norm(vec)
        vec_dir = vec / (vec_norm + 1e-6)
        vec_norm_limit = np.clip(vec_norm, -limit, limit)
        return vec_norm_limit * vec_dir

    def predict(
        self,
        acc_lin: np.ndarray,
        acc_ang: np.ndarray,
    ):
        trans_old_in_world = self.__obs_state.get_pose().get_trans()
        vel_lin = self.__obs_state.get_vel().get_linear()
        vel_ang = self.__obs_state.get_vel().get_angular()
        acc_lin = self.__norm_limit(acc_lin, self.__acc_lin_limit)
        acc_ang = self.__norm_limit(acc_ang, self.__acc_ang_limit)

        # runge-kutta k1
        dse3_1 = np.concatenate((vel_lin, vel_ang))
        ddse3_1 = np.concatenate((acc_lin, acc_ang))

        # runge-kutta k2
        dse3_2 = dse3_1 + ddse3_1 * self.__dt * 0.5
        ddse3_2 = ddse3_1

        # runge-kutta k3
        dse3_3 = dse3_1 + ddse3_2 * self.__dt * 0.5
        ddse3_3 = ddse3_1

        # runge-kutta k4
        dse3_4 = dse3_1 + ddse3_3 * self.__dt
        ddse3_4 = ddse3_1

        # runge-kutta
        se3_delta = (dse3_1 + 2.0 * dse3_2 + 2.0 * dse3_3 +
                     dse3_4) / 6.0 * self.__dt
        trans_old_in_new = se32trans(se3_delta)
        trans_new_in_world = trans_old_in_world @ trans_old_in_new
        pos_new, quat_new = trans2part(trans_new_in_world)
        vel_next = dse3_1 + (ddse3_1 + 2.0 * ddse3_2 + 2.0 * ddse3_3 +
                             ddse3_4) / 6.0 * self.__dt

        # clip
        vel_lin_next = self.__norm_limit(vel_next[:3], self.__vel_lin_limit)
        vel_ang_next = self.__norm_limit(vel_next[3:], self.__vel_ang_limit)

        # set state
        self.__obs_state.set_pose(HexCartPose(
            pos=pos_new,
            quat=quat_new,
        ))
        self.__obs_state.set_vel(
            HexCartVel(
                linear=vel_lin_next,
                angular=vel_ang_next,
            ))

    def update(self, state_sensor: HexCartState, update_weight: np.ndarray):
        pose_sensor = state_sensor.get_pose()
        pose_cur = self.__obs_state.get_pose()
        pos_sensor, quat_sensor = pose_sensor.get_pos(), pose_sensor.get_quat()
        pos_cur, quat_cur = pose_cur.get_pos(), pose_cur.get_quat()
        vel_sensor = state_sensor.get_vel()
        vel_cur = self.__obs_state.get_vel()
        vel_lin_sensor, vel_ang_sensor = vel_sensor.get_linear(
        ), vel_sensor.get_angular()
        vel_lin_cur, vel_ang_cur = vel_cur.get_linear(), vel_cur.get_angular()

        # update state
        obs_weight = 1.0 - update_weight
        pos_new = pos_cur * obs_weight[0] + pos_sensor * update_weight[0]
        quat_new = quat_slerp(quat_cur, quat_sensor, update_weight[1])
        vel_lin_new = vel_lin_cur * obs_weight[
            2] + vel_lin_sensor * update_weight[2]
        vel_ang_new = vel_ang_cur * obs_weight[
            3] + vel_ang_sensor * update_weight[3]
        self.__obs_state.set_pose(HexCartPose(pos_new, quat_new))
        self.__obs_state.set_vel(HexCartVel(vel_lin_new, vel_ang_new))
