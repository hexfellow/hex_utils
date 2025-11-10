#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-11-04
################################################################

import numpy as np


class HexCtrlUtilMitJoint:

    def __init__(self, ctrl_limit: np.ndarray | None = None):
        if ctrl_limit is not None:
            self.__ctrl_upper = ctrl_limit.copy()
            self.__ctrl_lower = -ctrl_limit.copy()
        else:
            self.__ctrl_upper = None
            self.__ctrl_lower = None

    def __call__(self, kp, kd, q_tar, dq_tar, q_cur, dq_cur, tau_comp):
        q_err = q_tar - q_cur
        dq_err = dq_tar - dq_cur
        tau_ctrl = kp * q_err + kd * dq_err
        if (self.__ctrl_upper is not None) and (self.__ctrl_lower is not None):
            tau_ctrl = np.clip(tau_ctrl, self.__ctrl_lower, self.__ctrl_upper)
        return tau_ctrl + tau_comp


class HexCtrlUtilPidJoint:

    def __init__(
        self,
        kp: np.ndarray,
        ki: np.ndarray,
        kd: np.ndarray,
        ctrl_limit: np.ndarray | None = None,
    ):
        self.__kp = kp.copy()
        self.__ki = ki.copy()
        self.__kd = kd.copy()
        if ctrl_limit is not None:
            self.__ctrl_upper = ctrl_limit.copy()
            self.__ctrl_lower = -ctrl_limit.copy()
        else:
            self.__ctrl_upper = None
            self.__ctrl_lower = None

        # buffer
        self.__last_ctrl = np.zeros_like(kp)
        self.__last_q_err = np.zeros_like(kp)
        self.__last_dq_err = np.zeros_like(kp)

    def __call__(self, q_tar, q_cur, dq_tar, dq_cur, tau_comp):
        q_err = q_tar - q_cur
        dq_err = dq_tar - dq_cur

        p_term = self.__kp * (q_err - self.__last_q_err)
        i_term = self.__ki * q_err
        d_term = self.__kd * (dq_err - self.__last_dq_err)
        delta_ctrl = p_term + i_term + d_term
        tau_ctrl = self.__last_ctrl + delta_ctrl
        if (self.__ctrl_upper is not None) and (self.__ctrl_lower is not None):
            tau_ctrl = np.clip(tau_ctrl, self.__ctrl_lower, self.__ctrl_upper)

        self.__last_ctrl = tau_ctrl
        self.__last_q_err = q_err
        self.__last_dq_err = dq_err
        return tau_ctrl + tau_comp


class HexCtrlUtilIntJoint:

    def __init__(
        self,
        ki,
        dt,
        limit=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        near_ratio=0.5,
        near_threshold=1e-1,
    ):
        # constants
        self.__ki = ki.copy()
        self.__dt = dt
        self.__limit_upper = np.array(
            [limit.copy(), limit.copy() * near_ratio])
        self.__limit_lower = np.array(
            [-limit.copy(), -limit.copy() * near_ratio])
        self.__near_threshold = near_threshold

        # variables
        self.__i_term = np.zeros(limit.shape[0])

    def __call__(self, cur_q, tar_q):
        err = tar_q - cur_q
        self.__i_term += self.__ki * err * self.__dt

        # limit idx
        col = np.arange(err.shape[0])
        row = (np.fabs(err) < self.__near_threshold).astype(int)

        # i_term limit
        self.__i_term = np.clip(
            self.__i_term,
            self.__limit_lower[row, col],
            self.__limit_upper[row, col],
        )
        return self.__i_term


class HexCtrlUtilMitWork:

    def __init__(self, ctrl_limit: np.ndarray | None = None):
        if ctrl_limit is not None:
            self.__ctrl_upper = ctrl_limit.copy()
            self.__ctrl_lower = -ctrl_limit.copy()
        else:
            self.__ctrl_upper = None
            self.__ctrl_lower = None

    def __call__(self, kp, kd, se3_tar, dse3_tar, se3_cur, dse3_cur, tau_comp):
        se3_err = se3_tar - se3_cur
        dse3_err = dse3_tar - dse3_cur
        tau_ctrl = kp * se3_err + kd * dse3_err
        if (self.__ctrl_upper is not None) and (self.__ctrl_lower is not None):
            tau_ctrl = np.clip(tau_ctrl, self.__ctrl_lower, self.__ctrl_upper)
        return tau_ctrl + tau_comp


class HexCtrlUtilIntWork:

    def __init__(
        self,
        ki,
        dt,
        limit=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        near_ratio=0.5,
        near_threshold=1e-1,
    ):
        # constants
        self.__ki = ki.copy()
        self.__dt = dt
        self.__limit_upper = np.array(
            [limit.copy(), limit.copy() * near_ratio])
        self.__limit_lower = np.array(
            [-limit.copy(), -limit.copy() * near_ratio])
        self.__near_threshold = near_threshold

        # variables
        self.__i_term = np.zeros(limit.shape[0])

    def __call__(self, cur_se3, tar_se3):
        se3_err = tar_se3 - cur_se3
        self.__i_term += self.__ki * se3_err * self.__dt

        # limit idx
        col = np.arange(se3_err.shape[0])
        row = (np.fabs(se3_err) < self.__near_threshold).astype(int)

        # i_term limit
        self.__i_term = np.clip(
            self.__i_term,
            self.__limit_lower[row, col],
            self.__limit_upper[row, col],
        )
        return self.__i_term
