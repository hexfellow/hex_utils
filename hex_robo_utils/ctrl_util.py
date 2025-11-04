#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-11-04
################################################################

import numpy as np


class HexCtrlUtilMit:

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


class HexCtrlUtilPid:

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
