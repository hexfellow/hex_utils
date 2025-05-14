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


class HexCartVel:

    def __init__(
            self,
            linear=np.zeros(3),
            angular=np.zeros(3),
    ):
        if self.__is_float_array(linear):
            self.__linear = copy.deepcopy(linear)
        else:
            raise TypeError(f"init linear type err: {type(linear)}")
        if self.__is_float_array(angular):
            self.__angular = copy.deepcopy(angular)
        else:
            raise TypeError(f"init angular type err: {type(angular)}")

    def __repr__(self):
        return f"linear: {self.__linear}; angular: {self.__angular}"

    def __is_float_array(self, element):
        return isinstance(element, np.ndarray) and isinstance(
            element[0], float)

    def linear(self):
        return self.__linear

    def angular(self):
        return self.__angular

    def get_linear(self):
        return copy.deepcopy(self.__linear)

    def get_angular(self):
        return copy.deepcopy(self.__angular)

    def set_linear(self, linear):
        if self.__is_float_array(linear):
            self.__linear = copy.deepcopy(linear)
        else:
            raise TypeError(f"set linear type err: {type(linear)}")

    def set_angular(self, angular):
        if self.__is_float_array(angular):
            self.__angular = copy.deepcopy(angular)
        else:
            raise TypeError(f"set angular type err: {type(angular)}")


class HexCartVelStamped:

    def __init__(
            self,
            stamp=HexStamp(),
            vel=HexCartVel(),
    ):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set stamp type err: {type(stamp)}")
        if self.__is_vel(vel):
            self.__vel = copy.deepcopy(vel)
        else:
            raise TypeError(f"set vel type err: {type(vel)}")

    def __repr__(self):
        print_str = f"stamp: {self.__stamp}\n"
        print_str += f"vel: {self.__vel}"
        return print_str

    def __is_stamp(self, stamp):
        return isinstance(stamp, HexStamp)

    def __is_vel(self, vel):
        return isinstance(vel, HexCartVel)

    def stamp(self):
        return self.__stamp

    def vel(self):
        return self.__vel

    def get_stamp(self):
        return copy.deepcopy(self.__stamp)

    def get_vel(self):
        return copy.deepcopy(self.__vel)

    def set_stamp(self, stamp):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set stamp type err: {type(stamp)}")

    def set_vel(self, vel):
        if self.__is_vel(vel):
            self.__vel = copy.deepcopy(vel)
        else:
            raise TypeError(f"set vel type err: {type(vel)}")


def main():
    stamp = HexStamp(1, 2)
    cart_vel_raw = HexCartVel(linear=np.zeros(3), angular=np.zeros(3))
    cart_vel_stamped_raw = HexCartVelStamped(stamp, cart_vel_raw)
    print(f"cart_vel_raw: {cart_vel_raw}")
    print(f"cart_vel_stamped_raw: {cart_vel_stamped_raw}")

    print("\n#### cart_vel copy ####")
    cart_vel_equal = None
    cart_vel_equal = cart_vel_raw
    cart_vel_copy = copy.copy(cart_vel_raw)
    cart_vel_deepcopy = copy.deepcopy(cart_vel_raw)
    cart_vel_raw.set_linear(np.ones(3) * 2.0)
    cart_vel_copy.set_linear(np.ones(3) * 3.0)
    print(f"cart_vel_raw: {cart_vel_raw}")
    print(f"cart_vel_equal: {cart_vel_equal}")
    print(f"cart_vel_copy: {cart_vel_copy}")
    print(f"cart_vel_deepcopy: {cart_vel_deepcopy}")

    print("\n#### cart_vel_stamped copy ####")
    cart_vel_stamped_equal = None
    cart_vel_stamped_equal = cart_vel_stamped_raw
    cart_vel_stamped_copy = copy.copy(cart_vel_stamped_raw)
    cart_vel_stamped_deepcopy = copy.deepcopy(cart_vel_stamped_raw)
    cart_vel_stamped_raw.stamp().set_time(3)
    cart_vel_stamped_raw.vel().set_linear(np.ones(3) * 2.0)
    print(f"cart_vel_stamped_raw: {cart_vel_stamped_raw}")
    print(f"cart_vel_stamped_equal: {cart_vel_stamped_equal}")
    print(f"cart_vel_stamped_copy: {cart_vel_stamped_copy}")
    print(f"cart_vel_stamped_deepcopy: {cart_vel_stamped_deepcopy}")


if __name__ == '__main__':
    main()
