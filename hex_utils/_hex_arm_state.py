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


class HexArmState:

    def __init__(
            self,
            pos=np.zeros(1),
            vel=np.ones(1) * 0.1,
            eff=np.ones(1) * 5.0,
    ):
        if self.__is_float_array(pos):
            self.__pos = copy.deepcopy(pos)
        else:
            raise TypeError(f"init pos type err: {type(pos)}")
        if self.__is_float_array(vel):
            self.__vel = copy.deepcopy(vel)
        else:
            raise TypeError(f"init vel type err: {type(vel)}")
        if self.__is_float_array(eff):
            self.__eff = copy.deepcopy(eff)
        else:
            raise TypeError(f"init eff type err: {type(eff)}")

        self.__joint_num = self.__pos.shape[0]

    def __repr__(self):
        return f"pos: {self.__pos}; vel: {self.__vel}; eff: {self.__eff}"

    def __is_float_array(self, element):
        return isinstance(element, np.ndarray) and isinstance(
            element[0], float)

    def get_joint_num(self):
        return self.__joint_num

    def pos(self):
        return self.__pos

    def vel(self):
        return self.__vel

    def eff(self):
        return self.__eff

    def get_pos(self):
        return copy.deepcopy(self.__pos)

    def get_vel(self):
        return copy.deepcopy(self.__vel)

    def get_eff(self):
        return copy.deepcopy(self.__eff)

    def set_pos(self, pos):
        if self.__is_float_array(pos):
            self.__pos = copy.deepcopy(pos)
        else:
            raise TypeError(f"set_pos type err: {type(pos)}")

    def set_vel(self, vel):
        if self.__is_float_array(vel):
            self.__vel = copy.deepcopy(vel)
        else:
            raise TypeError(f"set_vel type err: {type(vel)}")

    def set_eff(self, eff):
        if self.__is_float_array(eff):
            self.__eff = copy.deepcopy(eff)
        else:
            raise TypeError(f"set_eff type err: {type(eff)}")


class HexArmStateStamped:

    def __init__(
            self,
            stamp=HexStamp(),
            state=HexArmState(),
    ):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set_stamp type err: {type(stamp)}")
        if self.__is_state(state):
            self.__state = copy.deepcopy(state)
        else:
            raise TypeError(f"set_state type err: {type(state)}")

    def __repr__(self):
        print_str = f"stamp: {self.__stamp}\n"
        print_str += f"state: {self.__state}"
        return print_str

    def __is_stamp(self, stamp):
        return isinstance(stamp, HexStamp)

    def __is_state(self, state):
        return isinstance(state, HexArmState)

    def stamp(self):
        return self.__stamp

    def state(self):
        return self.__state

    def get_stamp(self):
        return copy.deepcopy(self.__stamp)

    def get_state(self):
        return copy.deepcopy(self.__state)

    def set_stamp(self, stamp):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set_stamp type err: {type(stamp)}")

    def set_state(self, state):
        if self.__is_state(state):
            self.__state = copy.deepcopy(state)
        else:
            raise TypeError(f"set_state type err: {type(state)}")


def main():
    stamp = HexStamp(1, 2)
    arm_state_raw = HexArmState(pos=np.zeros(6),
                                vel=np.ones(6) * 0.1,
                                eff=np.ones(6) * 5.0)
    arm_state_stamped_raw = HexArmStateStamped(stamp, arm_state_raw)
    print(f"arm_state_raw: {arm_state_raw}")
    print(f"arm_state_stamped_raw: {arm_state_stamped_raw}")

    print("\n#### arm_state copy ####")
    arm_state_equal = None
    arm_state_equal = arm_state_raw
    arm_state_copy = copy.copy(arm_state_raw)
    arm_state_deepcopy = copy.deepcopy(arm_state_raw)
    arm_state_raw.set_pos(np.ones(6))
    print(f"arm_state_raw: {arm_state_raw}")
    print(f"arm_state_equal: {arm_state_equal}")
    print(f"arm_state_copy: {arm_state_copy}")
    print(f"arm_state_deepcopy: {arm_state_deepcopy}")

    print("\n#### arm_state_stamped copy ####")
    arm_state_stamped_equal = None
    arm_state_stamped_equal = arm_state_stamped_raw
    arm_state_stamped_copy = copy.copy(arm_state_stamped_raw)
    arm_state_stamped_deepcopy = copy.deepcopy(arm_state_stamped_raw)
    arm_state_stamped_raw.stamp().set_time(3)
    arm_state_stamped_raw.state().set_pos(np.ones(6) * 2.0)
    print(f"arm_state_stamped_raw: {arm_state_stamped_raw}")
    print(f"arm_state_stamped_equal: {arm_state_stamped_equal}")
    print(f"arm_state_stamped_copy: {arm_state_stamped_copy}")
    print(f"arm_state_stamped_deepcopy: {arm_state_stamped_deepcopy}")


if __name__ == '__main__':
    main()
