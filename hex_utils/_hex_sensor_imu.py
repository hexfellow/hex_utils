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


class HexSensorImu:

    def __init__(
            self,
            acc=np.zeros(3),
            gyro=np.zeros(3),
    ):
        if self.__is_float_array(acc):
            self.__acc = copy.deepcopy(acc)
        else:
            raise TypeError(f"init acc type err: {type(acc)}")
        if self.__is_float_array(gyro):
            self.__gyro = copy.deepcopy(gyro)
        else:
            raise TypeError(f"init gyro type err: {type(gyro)}")

    def __repr__(self):
        print_str = f"acc: {self.__acc}\n"
        print_str += f"gyro: {self.__gyro}"
        return print_str

    def __is_float_array(self, element):
        return isinstance(element, np.ndarray) and isinstance(
            element[0], float)

    def acc(self):
        return self.__acc

    def gyro(self):
        return self.__gyro

    def get_acc(self):
        return copy.deepcopy(self.__acc)

    def get_gyro(self):
        return copy.deepcopy(self.__gyro)

    def set_acc(self, acc):
        if self.__is_float_array(acc):
            self.__acc = copy.deepcopy(acc)
        else:
            raise TypeError(f"set acc type err: {type(acc)}")

    def set_gyro(self, gyro):
        if self.__is_float_array(gyro):
            self.__gyro = copy.deepcopy(gyro)
        else:
            raise TypeError(f"set gyro type err: {type(gyro)}")


class HexSensorImuStamped:

    def __init__(
            self,
            stamp=HexStamp(),
            imu=HexSensorImu(),
    ):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set stamp type err: {type(stamp)}")
        if self.__is_imu(imu):
            self.__imu = copy.deepcopy(imu)
        else:
            raise TypeError(f"set imu type err: {type(imu)}")

    def __repr__(self):
        print_str = f"stamp: {self.__stamp}\n"
        print_str += f"imu: {self.__imu}"
        return print_str

    def __is_stamp(self, stamp):
        return isinstance(stamp, HexStamp)

    def __is_imu(self, imu):
        return isinstance(imu, HexSensorImu)

    def stamp(self):
        return self.__stamp

    def imu(self):
        return self.__imu

    def get_stamp(self):
        return copy.deepcopy(self.__stamp)

    def get_imu(self):
        return copy.deepcopy(self.__imu)

    def set_stamp(self, stamp):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set stamp type err: {type(stamp)}")

    def set_imu(self, imu):
        if self.__is_imu(imu):
            self.__imu = copy.deepcopy(imu)
        else:
            raise TypeError(f"set imu type err: {type(imu)}")

class HexSensorImuQuat:

    def __init__(
            self,
            acc=np.zeros(3),
            gyro=np.zeros(3),
            quat=np.zeros(4),
    ):
        if self.__is_float_array(acc):
            self.__acc = copy.deepcopy(acc)
        else:
            raise TypeError(f"init acc type err: {type(acc)}")
        if self.__is_float_array(gyro):
            self.__gyro = copy.deepcopy(gyro)
        else:
            raise TypeError(f"init gyro type err: {type(gyro)}")
        if self.__is_float_array(quat):
            self.__quat = copy.deepcopy(quat)
        else:
            raise TypeError(f"init quat type err: {type(quat)}")

    def __repr__(self):
        print_str = f"acc: {self.__acc}\n"
        print_str += f"gyro: {self.__gyro}\n"
        print_str += f"quat: {self.__quat}"
        return print_str

    def __is_float_array(self, element):
        return isinstance(element, np.ndarray) and isinstance(
            element[0], float)

    def acc(self):
        return self.__acc

    def gyro(self):
        return self.__gyro

    def quat(self):
        return self.__quat
    
    def get_acc(self):
        return copy.deepcopy(self.__acc)

    def get_gyro(self):
        return copy.deepcopy(self.__gyro)

    def get_quat(self):
        return copy.deepcopy(self.__quat)

    def set_acc(self, acc):
        if self.__is_float_array(acc):
            self.__acc = copy.deepcopy(acc)
        else:
            raise TypeError(f"set acc type err: {type(acc)}")

    def set_gyro(self, gyro):
        if self.__is_float_array(gyro):
            self.__gyro = copy.deepcopy(gyro)
        else:
            raise TypeError(f"set gyro type err: {type(gyro)}")
    
    def set_quat(self, quat):
        if self.__is_float_array(quat):
            self.__quat = copy.deepcopy(quat)
        else:
            raise TypeError(f"set quat type err: {type(quat)}")

class HexSensorImuQuatStamped:

    def __init__(
            self,
            stamp=HexStamp(),
            imu_quat=HexSensorImuQuat(),
    ):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set stamp type err: {type(stamp)}")
        if self.__is_imu_quat(imu_quat):
            self.__imu_quat = copy.deepcopy(imu_quat)
        else:
            raise TypeError(f"set imu_quat type err: {type(imu_quat)}")

    def __repr__(self):
        print_str = f"stamp: {self.__stamp}\n"
        print_str += f"imu_quat: {self.__imu_quat}"
        return print_str

    def __is_stamp(self, stamp):
        return isinstance(stamp, HexStamp)

    def __is_imu_quat(self, imu_quat):
        return isinstance(imu_quat, HexSensorImuQuat)

    def stamp(self):
        return self.__stamp

    def imu_quat(self):
        return self.__imu_quat

    def get_stamp(self):
        return copy.deepcopy(self.__stamp)

    def get_imu_quat(self):
        return copy.deepcopy(self.__imu_quat)

    def set_stamp(self, stamp):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set stamp type err: {type(stamp)}")

    def set_imu_quat(self, imu_quat):
        if self.__is_imu_quat(imu_quat):
            self.__imu_quat = copy.deepcopy(imu_quat)
        else:
            raise TypeError(f"set imu_quat type err: {type(imu_quat)}")


def main():
    stamp = HexStamp(1, 2)
    acc = np.array([1.0, 2.0, 3.0])
    gyro = np.array([1.0, 2.0, 3.0])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    imu_raw = HexSensorImu(acc, gyro)
    imu_stamped_raw = HexSensorImuStamped(stamp, imu_raw)
    imu_quat_raw = HexSensorImuQuat(acc, gyro, quat)
    imu_quat_stamped_raw = HexSensorImuQuatStamped(stamp, imu_quat_raw)
    print(f"imu_raw: {imu_raw}")
    print(f"imu_stamped_raw: {imu_stamped_raw}")
    print(f"imu_quat_raw: {imu_quat_raw}")
    print(f"imu_quat_stamped_raw: {imu_quat_stamped_raw}")

    print("\n#### imu copy ####")
    imu_equal = None
    imu_equal = imu_raw
    imu_copy = copy.copy(imu_raw)
    imu_deepcopy = copy.deepcopy(imu_raw)
    imu_raw.set_acc(np.ones(3) * 2.0)
    imu_raw.set_gyro(np.ones(3) * 3.0)
    print(f"imu_raw: {imu_raw}")
    print(f"imu_equal: {imu_equal}")
    print(f"imu_copy: {imu_copy}")
    print(f"imu_deepcopy: {imu_deepcopy}")
    
    print("\n#### imu_quat copy ####")
    imu_quat_equal = None
    imu_quat_equal = imu_quat_raw
    imu_quat_copy = copy.copy(imu_quat_raw)
    imu_quat_deepcopy = copy.deepcopy(imu_quat_raw)
    imu_quat_raw.set_acc(np.ones(3) * 2.0)
    imu_quat_raw.set_gyro(np.ones(3) * 3.0)
    imu_quat_raw.set_quat(np.ones(4) * 1.0)
    print(f"imu_quat_raw: {imu_quat_raw}")
    print(f"imu_quat_equal: {imu_quat_equal}")
    print(f"imu_quat_copy: {imu_quat_copy}")
    print(f"imu_quat_deepcopy: {imu_quat_deepcopy}")

    print("\n#### imu_stamped copy ####")
    imu_stamped_equal = None
    imu_stamped_equal = imu_stamped_raw
    imu_stamped_copy = copy.copy(imu_stamped_raw)
    imu_stamped_deepcopy = copy.deepcopy(imu_stamped_raw)
    imu_stamped_raw.stamp().set_time(3)
    imu_stamped_raw.imu().set_acc(np.ones(3) * 2.0)
    imu_stamped_raw.imu().set_gyro(np.ones(3) * 3.0)
    print(f"imu_stamped_raw: {imu_stamped_raw}")
    print(f"imu_stamped_equal: {imu_stamped_equal}")
    print(f"imu_stamped_copy: {imu_stamped_copy}")
    print(f"imu_stamped_deepcopy: {imu_stamped_deepcopy}")

    print("\n#### imu_quat_stamped copy ####")
    imu_quat_stamped_equal = None
    imu_quat_stamped_equal = imu_quat_stamped_raw
    imu_quat_stamped_copy = copy.copy(imu_quat_stamped_raw)     
    imu_quat_stamped_deepcopy = copy.deepcopy(imu_quat_stamped_raw)
    imu_quat_stamped_raw.stamp().set_time(3)
    imu_quat_stamped_raw.imu_quat().set_acc(np.ones(3) * 2.0)
    imu_quat_stamped_raw.imu_quat().set_gyro(np.ones(3) * 3.0)
    imu_quat_stamped_raw.imu_quat().set_quat(np.ones(4) * 1.0)
    print(f"imu_quat_stamped_raw: {imu_quat_stamped_raw}")
    print(f"imu_quat_stamped_equal: {imu_quat_stamped_equal}")
    print(f"imu_quat_stamped_copy: {imu_quat_stamped_copy}")
    print(f"imu_quat_stamped_deepcopy: {imu_quat_stamped_deepcopy}")

if __name__ == "__main__":
    main()
