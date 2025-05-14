#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-04-21
################################################################

import copy
import numpy as np

from hex_utils._hex_stamp import HexStamp
from hex_utils._hex_cart_pose import HexCartPose
from hex_utils._hex_cart_vel import HexCartVel


class HexCartState:

    def __init__(
            self,
            pose=HexCartPose(),
            vel=HexCartVel(),
    ):
        if self.__is_pose(pose):
            self.__pose = copy.deepcopy(pose)
        else:
            raise TypeError(f"init pose type err: {type(pose)}")
        if self.__is_vel(vel):
            self.__vel = copy.deepcopy(vel)
        else:
            raise TypeError(f"init vel type err: {type(vel)}")

    def __repr__(self):
        print_str = f"pose: {self.__pose}\n"
        print_str += f"vel: {self.__vel}"
        return print_str

    def __is_pose(self, pose):
        return isinstance(pose, HexCartPose)

    def __is_vel(self, vel):
        return isinstance(vel, HexCartVel)

    def pose(self):
        return self.__pose

    def vel(self):
        return self.__vel

    def get_pose(self):
        return copy.deepcopy(self.__pose)

    def get_vel(self):
        return copy.deepcopy(self.__vel)

    def set_pose(self, pose):
        if self.__is_pose(pose):
            self.__pose = copy.deepcopy(pose)
        else:
            raise TypeError(f"set pose type err: {type(pose)}")

    def set_vel(self, vel):
        if self.__is_vel(vel):
            self.__vel = copy.deepcopy(vel)
        else:
            raise TypeError(f"set vel type err: {type(vel)}")


class HexCartStateStamped:

    def __init__(
            self,
            stamp=HexStamp(),
            state=HexCartState(),
    ):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"init stamp type err: {type(stamp)}")
        if self.__is_state(state):
            self.__state = copy.deepcopy(state)
        else:
            raise TypeError(f"init state type err: {type(state)}")

    def __repr__(self):
        print_str = f"stamp: {self.__stamp}\n"
        print_str += f"state: {self.__state}"
        return print_str

    def __is_stamp(self, stamp):
        return isinstance(stamp, HexStamp)

    def __is_state(self, state):
        return isinstance(state, HexCartState)

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
            raise TypeError(f"set stamp type err: {type(stamp)}")

    def set_state(self, state):
        if self.__is_state(state):
            self.__state = copy.deepcopy(state)
        else:
            raise TypeError(f"set state type err: {type(state)}")


def main():
    stamp = HexStamp(1, 2)
    cart_pose_raw = HexCartPose(pos=np.zeros(3),
                                quat=np.array([1.0, 0.0, 0.0, 0.0]))
    cart_vel_raw = HexCartVel(linear=np.zeros(3), angular=np.zeros(3))
    cart_state_raw = HexCartState(cart_pose_raw, cart_vel_raw)
    cart_state_stamped_raw = HexCartStateStamped(stamp, cart_state_raw)
    print(f"cart_state_raw: {cart_state_raw}")
    print(f"cart_state_stamped_raw: {cart_state_stamped_raw}")

    print("\n#### cart_state copy ####")
    cart_state_equal = None
    cart_state_equal = cart_state_raw
    cart_state_copy = copy.copy(cart_state_raw)
    cart_state_deepcopy = copy.deepcopy(cart_state_raw)
    cart_state_raw.pose().set_pos(np.ones(3))
    cart_state_raw.vel().set_linear(np.ones(3))
    print(f"cart_state_raw: {cart_state_raw}")
    print(f"cart_state_equal: {cart_state_equal}")
    print(f"cart_state_copy: {cart_state_copy}")
    print(f"cart_state_deepcopy: {cart_state_deepcopy}")

    print("\n#### cart_state_stamped copy ####")
    cart_state_stamped_equal = None
    cart_state_stamped_equal = cart_state_stamped_raw
    cart_state_stamped_copy = copy.copy(cart_state_stamped_raw)
    cart_state_stamped_deepcopy = copy.deepcopy(cart_state_stamped_raw)
    cart_state_stamped_raw.stamp().set_time(3)
    cart_state_stamped_raw.state().pose().set_pos(np.ones(3))
    cart_state_stamped_raw.state().vel().set_linear(np.ones(3))
    print(f"cart_state_stamped_raw: {cart_state_stamped_raw}")
    print(f"cart_state_stamped_equal: {cart_state_stamped_equal}")
    print(f"cart_state_stamped_copy: {cart_state_stamped_copy}")
    print(f"cart_state_stamped_deepcopy: {cart_state_stamped_deepcopy}")


if __name__ == '__main__':
    main()
