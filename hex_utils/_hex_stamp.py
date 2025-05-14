#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-04-19
################################################################

import copy
import numpy as np


class HexStamp:

    def __init__(self, element_1=0, element_2=0):
        if self.__is_stamp(element_1):
            self.__sec = element_1.__sec
            self.__nsec = element_1.__nsec
        elif self.__is_sec(element_1) and self.__is_nsec(element_2):
            self.__sec = np.int64(element_1)
            self.__nsec = np.uint32(element_2)
        else:
            raise ValueError(
                f"Invalid type: {type(element_1)}, {type(element_2)}")

    def __repr__(self):
        return f"sec: {self.__sec}; nsec: {self.__nsec}"

    def get_time(self):
        return self.__sec, self.__nsec

    def get_time_float(self):
        return self.__sec + self.__nsec * 1e-9

    def set_time(self, sec, nsec=None):
        if self.__is_sec(sec):
            self.__sec = np.int64(sec)
        else:
            raise ValueError(f"Invalid type: {type(sec)}")

        if self.__is_nsec(nsec):
            self.__nsec = np.uint32(nsec)
        elif nsec is not None:
            raise ValueError(f"Invalid type: {type(nsec)}")

    def set_time_float(self, time_float: np.float64):
        if self.__is_float(time_float):
            if time_float < 0:
                raise ValueError(f"Invalid time: {time_float}")
            self.__sec = np.int64(time_float)
            self.__nsec = np.uint32((time_float - self.__sec) * 1e9)
        else:
            raise ValueError(f"Invalid type: {type(time_float)}")

    def __is_stamp(self, element):
        return isinstance(element, HexStamp)

    def __is_sec(self, element):
        return isinstance(element, int) or isinstance(element, np.int64)

    def __is_nsec(self, element):
        return isinstance(element, int) or isinstance(element, np.uint32)

    def __is_float(self, element):
        return isinstance(element, float)

    def __lt__(self, other: 'HexStamp'):
        if self.__sec < other.__sec:
            return True
        elif self.__sec == other.__sec:
            return self.__nsec < other.__nsec
        else:
            return False

    def __gt__(self, other: 'HexStamp'):
        if self.__sec > other.__sec:
            return True
        elif self.__sec == other.__sec:
            return self.__nsec > other.__nsec
        else:
            return False

    def __eq__(self, other: 'HexStamp'):
        return self.__sec == other.__sec and self.__nsec == other.__nsec

    def __ne__(self, other: 'HexStamp'):
        return not self.__eq__(other)

    def __le__(self, other: 'HexStamp'):
        return not self.__gt__(other)

    def __ge__(self, other: 'HexStamp'):
        return not self.__lt__(other)

    def __add__(self, delta: np.float64):
        delta_sec = np.int64(delta)
        delta_nsec = np.int64((delta - delta_sec) * 1e9)

        result_sec = self.__sec + delta_sec
        result_nsec = np.int64(self.__nsec + delta_nsec)
        if result_nsec >= 1e9:
            result_sec += 1
            result_nsec -= np.int64(1e9)
        elif result_nsec < 0:
            result_sec -= 1
            result_nsec += np.int64(1e9)
        return HexStamp(np.int64(result_sec), np.uint32(result_nsec))

    def __sub__(self, other):
        if self.__is_float(other):
            return self.__add__(-other)
        elif self.__is_stamp(other):
            delta_sec = float(self.__sec) - float(other.__sec)
            delta_nsec = float(self.__nsec) - float(other.__nsec)
            return delta_sec + delta_nsec * 1e-9
        else:
            raise ValueError(f"Invalid type: {type(other)}")


def main():
    stamp1 = HexStamp(1, 2)
    stamp2 = HexStamp(2, 3)
    print(f"stamp1: {stamp1}")
    print(f"stamp2: {stamp2}")

    print("\n#### copy ####")
    stamp_equal = None
    stamp_equal = stamp1
    stamp3 = HexStamp(stamp1)
    stamp4 = copy.copy(stamp1)
    stamp5 = copy.deepcopy(stamp1)
    stamp1.set_time(3)
    print(f"stamp1: {stamp1}")
    print(f"stamp_equal: {stamp_equal}")
    print(f"stamp3: {stamp3}")
    print(f"stamp4: {stamp4}")
    print(f"stamp5: {stamp5}")

    print("\n#### compare ####")
    print(f"stamp1: {stamp1}")
    print(f"stamp2: {stamp2}")
    if stamp1 < stamp2:
        print("stamp1 < stamp2")
    else:
        print("stamp1 >= stamp2")
    print(f"stamp3: {stamp3}")
    print(f"stamp4: {stamp4}")
    if stamp3 == stamp4:
        print("stamp3 == stamp4")
    else:
        print("stamp3 != stamp4")

    print("\n#### add ####")
    print(f"stamp1: {stamp1}")
    stamp6 = stamp1 + 1.0
    print(f"stamp6 = stamp1 + 1.0")
    print(f"stamp6: {stamp6}")

    print("\n#### sub ####")
    print(f"stamp2: {stamp2}")
    stamp7 = stamp2 - 1.0
    print(f"stamp7 = stamp2 - 1.0")
    print(f"stamp7: {stamp7}")
    print(f"stamp7 - stamp2: {stamp7 - stamp2}")


if __name__ == '__main__':
    main()
