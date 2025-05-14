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


class HexCartPose:

    def __init__(
            self,
            pos=np.zeros(3),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
    ):
        if self.__is_float_array(pos):
            self.__pos = copy.deepcopy(pos)
        else:
            raise TypeError(f"init pos type err: {type(pos)}")
        if self.__is_float_array(quat):
            self.__quat = copy.deepcopy(quat)
        else:
            raise TypeError(f"init quat type err: {type(quat)}")

    def __repr__(self):
        return f"pos: {self.__pos}; quat: {self.__quat}"

    def __is_float_array(self, element):
        return isinstance(element, np.ndarray) and isinstance(
            element[0], float)

    def pos(self):
        return self.__pos

    def quat(self):
        return self.__quat

    def get_pos(self):
        return copy.deepcopy(self.__pos)

    def get_quat(self):
        return copy.deepcopy(self.__quat)

    def set_pos(self, pos):
        if self.__is_float_array(pos):
            self.__pos = copy.deepcopy(pos)
        else:
            raise TypeError(f"set pos type err: {type(pos)}")

    def set_quat(self, quat):
        if self.__is_float_array(quat):
            self.__quat = copy.deepcopy(quat)
        else:
            raise TypeError(f"set quat type err: {type(quat)}")


class HexCartPoseStamped:

    def __init__(
            self,
            stamp=HexStamp(),
            pose=HexCartPose(),
    ):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"init stamp type err: {type(stamp)}")
        if self.__is_pose(pose):
            self.__pose = copy.deepcopy(pose)
        else:
            raise TypeError(f"init pose type err: {type(pose)}")

    def __repr__(self):
        print_str = f"stamp: {self.__stamp}\n"
        print_str += f"pose: {self.__pose}"
        return print_str

    def __is_stamp(self, stamp):
        return isinstance(stamp, HexStamp)

    def __is_pose(self, pose):
        return isinstance(pose, HexCartPose)

    def stamp(self):
        return self.__stamp

    def pose(self):
        return self.__pose

    def get_stamp(self):
        return copy.deepcopy(self.__stamp)

    def get_pose(self):
        return copy.deepcopy(self.__pose)

    def set_stamp(self, stamp):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set stamp type err: {type(stamp)}")

    def set_pose(self, pose):
        if self.__is_pose(pose):
            self.__pose = copy.deepcopy(pose)
        else:
            raise TypeError(f"set pose type err: {type(pose)}")


def main():
    stamp = HexStamp(1, 2)
    cart_pose_raw = HexCartPose(pos=np.zeros(3),
                                quat=np.array([1.0, 0.0, 0.0, 0.0]))
    cart_pose_stamped_raw = HexCartPoseStamped(stamp, cart_pose_raw)
    print(f"cart_pose_raw: {cart_pose_raw}")
    print(f"cart_pose_stamped_raw: {cart_pose_stamped_raw}")

    print("\n#### cart_pose copy ####")
    cart_pose_equal = None
    cart_pose_equal = cart_pose_raw
    cart_pose_copy = copy.copy(cart_pose_raw)
    cart_pose_deepcopy = copy.deepcopy(cart_pose_raw)
    cart_pose_raw.set_pos(np.ones(3))
    print(f"cart_pose_raw: {cart_pose_raw}")
    print(f"cart_pose_equal: {cart_pose_equal}")
    print(f"cart_pose_copy: {cart_pose_copy}")
    print(f"cart_pose_deepcopy: {cart_pose_deepcopy}")

    print("\n#### cart_pose_stamped copy ####")
    cart_pose_stamped_equal = None
    cart_pose_stamped_equal = cart_pose_stamped_raw
    cart_pose_stamped_copy = copy.copy(cart_pose_stamped_raw)
    cart_pose_stamped_deepcopy = copy.deepcopy(cart_pose_stamped_raw)
    cart_pose_stamped_raw.stamp().set_time(3)
    cart_pose_stamped_raw.pose().set_pos(np.ones(3) * 2.0)
    print(f"cart_pose_stamped_raw: {cart_pose_stamped_raw}")
    print(f"cart_pose_stamped_equal: {cart_pose_stamped_equal}")
    print(f"cart_pose_stamped_copy: {cart_pose_stamped_copy}")
    print(f"cart_pose_stamped_deepcopy: {cart_pose_stamped_deepcopy}")


if __name__ == '__main__':
    main()
