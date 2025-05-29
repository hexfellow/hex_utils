#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-05-29
################################################################

import numpy as np

from hex_utils._hex_arm_state import HexArmState


class ObsUtil:

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
        ddq1 = self.__ddq(q_tar, q_cur, dq_cur)

        # runge-kutta k2
        q2 = q_cur + dq1 * self.__dt * 0.5
        dq2 = dq_cur + ddq1 * self.__dt * 0.5
        ddq2 = self.__ddq(q_tar, q2, dq2)

        # runge-kutta k3
        q3 = q_cur + dq2 * self.__dt * 0.5
        dq3 = dq_cur + ddq2 * self.__dt * 0.5
        ddq3 = self.__ddq(q_tar, q3, dq3)

        # runge-kutta k4
        q4 = q_cur + dq3 * self.__dt
        dq4 = dq_cur + ddq3 * self.__dt
        ddq4 = self.__ddq(q_tar, q4, dq4)

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

    def __ddq(self, q_tar: np.ndarray, q_cur: np.ndarray,
              dq_cur: np.ndarray) -> np.ndarray:
        ddq = (self.__stiff @ (q_tar - q_cur) -
               self.__damp @ dq_cur) @ self.__mass_inv
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
