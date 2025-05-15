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

class HexSensorMag:
    def __init__(
            self,
            magnetic_field=np.zeros(3),
    ):
        if self.__is_float_array(magnetic_field):
            self.__magnetic_field = copy.deepcopy(magnetic_field)
        else:
            raise TypeError(f"init magnetic_field type err: {type(magnetic_field)}")

    def __repr__(self):
        print_str = f"magnetic_field: {self.__magnetic_field}\n"
        return print_str

    def __is_float_array(self, element):
        return isinstance(element, np.ndarray) and isinstance(
            element[0], float)

    def magnetic_field(self):
        return self.__magnetic_field
    
    def get_magnetic_field(self):
        return copy.deepcopy(self.__magnetic_field)

    def set_magnetic_field(self, magnetic_field):
        if self.__is_float_array(magnetic_field):
            self.__magnetic_field = copy.deepcopy(magnetic_field)
        else:
            raise TypeError(f"set magnetic_field type err: {type(magnetic_field)}")

class HexSensorMagStamped:

    def __init__(
            self,
            stamp=HexStamp(),
            magnetic_field=HexSensorMag(),
    ):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set stamp type err: {type(stamp)}")
        if self.__is_magnetic_field(magnetic_field):
            self.__magnetic_field = copy.deepcopy(magnetic_field)
        else:
            raise TypeError(f"set magnetic_field type err: {type(magnetic_field)}")

    def __repr__(self):
        print_str = f"stamp: {self.__stamp}\n"
        print_str += f"magnetic_field: {self.__magnetic_field}"
        return print_str

    def __is_stamp(self, stamp):
        return isinstance(stamp, HexStamp)

    def __is_magnetic_field(self, magnetic_field):
        return isinstance(magnetic_field, HexSensorMag)

    def stamp(self):
        return self.__stamp

    def magnetic_field(self):
        return self.__magnetic_field

    def get_stamp(self):
        return copy.deepcopy(self.__stamp)

    def get_magnetic_field(self):
        return copy.deepcopy(self.__magnetic_field)

    def set_stamp(self, stamp):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set stamp type err: {type(stamp)}")

    def set_magnetic_field(self, magnetic_field):
        if self.__is_magnetic_field(magnetic_field):
            self.__magnetic_field = copy.deepcopy(magnetic_field)
        else:
            raise TypeError(f"set magnetic_field type err: {type(magnetic_field)}")

def main():
    stamp = HexStamp(1, 2)
    magnetic_field = np.array([1.0, 2.0, 3.0])
    magnetic_field_raw = HexSensorMag(magnetic_field)
    magnetic_field_stamped_raw = HexSensorMagStamped(stamp, magnetic_field_raw)
    print(f"magnetic_field_raw: {magnetic_field_raw}")
    print(f"magnetic_field_stamped_raw: {magnetic_field_stamped_raw}")

    print("\n#### magnetic_field copy ####")
    magnetic_field_equal = None
    magnetic_field_equal = magnetic_field_raw
    magnetic_field_copy = copy.copy(magnetic_field_raw)
    magnetic_field_deepcopy = copy.deepcopy(magnetic_field_raw)
    magnetic_field_raw.set_magnetic_field(np.ones(3) * 2.0)
    print(f"magnetic_field_raw: {magnetic_field_raw}")
    print(f"magnetic_field_equal: {magnetic_field_equal}")
    print(f"magnetic_field_copy: {magnetic_field_copy}")
    print(f"magnetic_field_deepcopy: {magnetic_field_deepcopy}")

    print("\n#### magnetic_field_stamped copy ####")
    magnetic_field_stamped_equal = None
    magnetic_field_stamped_equal = magnetic_field_stamped_raw
    magnetic_field_stamped_copy = copy.copy(magnetic_field_stamped_raw)
    magnetic_field_stamped_deepcopy = copy.deepcopy(magnetic_field_stamped_raw)
    magnetic_field_stamped_raw.stamp().set_time(3)
    magnetic_field_stamped_raw.magnetic_field().set_magnetic_field(np.ones(3) * 2.0)
    print(f"magnetic_field_stamped_raw: {magnetic_field_stamped_raw}")
    print(f"magnetic_field_stamped_equal: {magnetic_field_stamped_equal}")
    print(f"magnetic_field_stamped_copy: {magnetic_field_stamped_copy}")
    print(f"magnetic_field_stamped_deepcopy: {magnetic_field_stamped_deepcopy}")


if __name__ == "__main__":
    main()

