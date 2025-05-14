#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-04-19
################################################################

import copy
import numpy as np

from hex_utils._hex_stamp import HexStamp
from hex_utils._hex_arm_state import HexArmState


class HexArmCtrl:

    def __init__(
            self,
            mode=["position_mode"] * 1,
            param=[{
                "braking_state": False
            }] * 1,
            state=HexArmState(),
    ):
        ### get input data
        mode_len = param_len = state_len = 0

        # mode
        if self.__is_mode(mode):
            mode_len = len(mode)
            self.__mode = copy.deepcopy(mode)
        else:
            print(f"mode type err. Use default mode.")
        # param
        if self.__is_param(param):
            param_len = len(param)
            self.__param = copy.deepcopy(param)
        else:
            print(f"param type err. Use default param.")
        # state
        if self.__is_state(state):
            state_len = state.get_joint_num()
            self.__state = copy.deepcopy(state)
        else:
            print(f"state type err. Use default state.")

        # check joint_num
        joint_num = max(mode_len, param_len, state_len)
        if joint_num == 0:
            print("No feasible data input. Use default value.")
            joint_num = 6

        ### set default data
        if mode_len != joint_num:
            self.__mode = ["position_mode"] * joint_num
        if param_len != joint_num:
            self.__param = [copy.deepcopy({"braking_state": False})
                            ] * joint_num
        if state_len != joint_num:
            self.__state = HexArmState(
                pos=np.zeros(joint_num),
                vel=np.ones(joint_num) * 0.1,
                eff=np.ones(joint_num) * 5.0,
            )

    def __repr__(self):
        print_str = f"mode: {self.__mode}\n"
        print_str += f"param: {self.__param}\n"
        print_str += f"state: {self.__state}"
        return print_str

    def __is_mode(self, mode):
        return isinstance(mode, list) and isinstance(mode[0], str)

    def __is_param(self, param):
        return isinstance(param, list) and isinstance(param[0], dict)

    def __is_state(self, state):
        return isinstance(state, HexArmState)

    def get_joint_num(self):
        return self.__state.get_joint_num()

    def mode(self):
        return self.__mode

    def param(self):
        return self.__param

    def state(self):
        return self.__state

    def get_mode(self):
        return copy.deepcopy(self.__mode)

    def get_param(self):
        return copy.deepcopy(self.__param)

    def get_state(self):
        return copy.deepcopy(self.__state)

    def set_mode(self, mode):
        if self.__is_mode(mode):
            self.__mode = copy.deepcopy(mode)
        else:
            raise TypeError(f"set_mode type err: {type(mode)}")

    def set_param(self, param):
        if self.__is_param(param):
            self.__param = copy.deepcopy(param)
        else:
            raise TypeError(f"set_param type err: {type(param)}")

    def set_state(self, state):
        if self.__is_state(state):
            self.__state = copy.deepcopy(state)
        else:
            raise TypeError(f"set_state type err: {type(state)}")


class HexArmCtrlStamped:

    def __init__(
            self,
            stamp=HexStamp(),
            ctrl=HexArmCtrl(),
    ):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set_stamp type err: {type(stamp)}")

        if self.__is_ctrl(ctrl):
            self.__ctrl = copy.deepcopy(ctrl)
        else:
            raise TypeError(f"set_ctrl type err: {type(ctrl)}")

    def __repr__(self):
        print_str = f"stamp: {self.__stamp}\n"
        print_str += f"ctrl: {self.__ctrl}"
        return print_str

    def __is_stamp(self, stamp):
        return isinstance(stamp, HexStamp)

    def __is_ctrl(self, ctrl):
        return isinstance(ctrl, HexArmCtrl)

    def stamp(self):
        return self.__stamp

    def ctrl(self):
        return self.__ctrl

    def get_stamp(self):
        return copy.deepcopy(self.__stamp)

    def get_ctrl(self):
        return copy.deepcopy(self.__ctrl)

    def set_stamp(self, stamp):
        if self.__check_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set_stamp type err: {type(stamp)}")

    def set_ctrl(self, ctrl):
        if self.__check_ctrl(ctrl):
            self.__ctrl = copy.deepcopy(ctrl)
        else:
            raise TypeError(f"set_ctrl type err: {type(ctrl)}")


def main():
    stamp = HexStamp(1, 2)
    mode = ["position_mode"] * 6
    param = [{"braking_state": False}] * 6
    state = HexArmState(pos=np.zeros(6),
                        vel=np.ones(6) * 0.1,
                        eff=np.ones(6) * 5.0)
    arm_ctrl_raw = HexArmCtrl(mode, param, state)
    arm_ctrl_stamped_raw = HexArmCtrlStamped(stamp, arm_ctrl_raw)
    print(f"arm_ctrl_raw: {arm_ctrl_raw}")
    print(f"arm_ctrl_stamped_raw: {arm_ctrl_stamped_raw}")

    print("\n#### arm_ctrl copy ####")
    arm_ctrl_equal = None
    arm_ctrl_equal = arm_ctrl_raw
    arm_ctrl_copy = copy.copy(arm_ctrl_raw)
    arm_ctrl_deepcopy = copy.deepcopy(arm_ctrl_raw)
    arm_ctrl_raw.set_mode(["mit_mode"] * 6)
    arm_ctrl_raw.set_param([{"braking_state": True}] * 6)
    print(f"arm_ctrl_raw: {arm_ctrl_raw}")
    print(f"arm_ctrl_equal: {arm_ctrl_equal}")
    print(f"arm_ctrl_copy: {arm_ctrl_copy}")
    print(f"arm_ctrl_deepcopy: {arm_ctrl_deepcopy}")

    print("\n#### arm_ctrl_stamped copy ####")
    arm_ctrl_stamped_equal = None
    arm_ctrl_stamped_equal = arm_ctrl_stamped_raw
    arm_ctrl_stamped_copy = copy.copy(arm_ctrl_stamped_raw)
    arm_ctrl_stamped_deepcopy = copy.deepcopy(arm_ctrl_stamped_raw)
    arm_ctrl_stamped_raw.stamp().set_time(3)
    arm_ctrl_stamped_raw.ctrl().set_mode(["mit_mode"] * 6)
    arm_ctrl_stamped_raw.ctrl().set_param([{"braking_state": True}] * 6)
    arm_ctrl_stamped_raw.ctrl().state().set_pos(np.ones(6))
    print(f"arm_ctrl_stamped_raw: {arm_ctrl_stamped_raw}")
    print(f"arm_ctrl_stamped_equal: {arm_ctrl_stamped_equal}")
    print(f"arm_ctrl_stamped_copy: {arm_ctrl_stamped_copy}")
    print(f"arm_ctrl_stamped_deepcopy: {arm_ctrl_stamped_deepcopy}")


if __name__ == '__main__':
    main()
