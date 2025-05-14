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


class HexSensorScan:

    def __init__(
            self,
            distances=np.zeros(1),
            intensities=np.zeros(1),
    ):
        if self.__is_float_array(distances):
            self.__distances = copy.deepcopy(distances)
        else:
            raise TypeError(f"init distances type err: {type(distances)}")
        if self.__is_float_array(intensities):
            self.__intensities = copy.deepcopy(intensities)
        else:
            raise TypeError(f"init intensities type err: {type(intensities)}")

    def __repr__(self):
        print_str = f"distances: {self.__distances}\n"
        print_str += f"intensities: {self.__intensities}"
        return print_str

    def __is_float_array(self, element):
        return isinstance(element, np.ndarray) and isinstance(
            element[0], float)

    def distances(self):
        return self.__distances

    def intensities(self):
        return self.__intensities

    def get_distances(self):
        return copy.deepcopy(self.__distances)

    def get_intensities(self):
        return copy.deepcopy(self.__intensities)

    def set_distances(self, distances):
        if self.__is_float_array(distances):
            self.__distances = copy.deepcopy(distances)
        else:
            raise TypeError(f"set distances type err: {type(distances)}")

    def set_intensities(self, intensities):
        if self.__is_float_array(intensities):
            self.__intensities = copy.deepcopy(intensities)
        else:
            raise TypeError(f"set intensities type err: {type(intensities)}")


class HexSensorScanStamped:

    def __init__(
            self,
            stamp=HexStamp(),
            scan=HexSensorScan(),
    ):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set stamp type err: {type(stamp)}")
        if self.__is_scan(scan):
            self.__scan = copy.deepcopy(scan)
        else:
            raise TypeError(f"set scan type err: {type(scan)}")

    def __repr__(self):
        print_str = f"stamp: {self.__stamp}\n"
        print_str += f"scan: {self.__scan}"
        return print_str

    def __is_stamp(self, stamp):
        return isinstance(stamp, HexStamp)

    def __is_scan(self, scan):
        return isinstance(scan, HexSensorScan)

    def stamp(self):
        return self.__stamp

    def scan(self):
        return self.__scan

    def get_stamp(self):
        return copy.deepcopy(self.__stamp)

    def get_scan(self):
        return copy.deepcopy(self.__scan)

    def set_stamp(self, stamp):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set stamp type err: {type(stamp)}")

    def set_scan(self, scan):
        if self.__is_scan(scan):
            self.__scan = copy.deepcopy(scan)
        else:
            raise TypeError(f"set scan type err: {type(scan)}")


def main():
    stamp = HexStamp(1, 2)
    distances = np.array([1.0, 2.0, 3.0])
    intensities = np.array([1.0, 2.0, 3.0])
    scan_raw = HexSensorScan(distances, intensities)
    scan_stamped_raw = HexSensorScanStamped(stamp, scan_raw)
    print(f"scan_raw: {scan_raw}")
    print(f"scan_stamped_raw: {scan_stamped_raw}")

    print("\n#### scan copy ####")
    scan_equal = None
    scan_equal = scan_raw
    scan_copy = copy.copy(scan_raw)
    scan_deepcopy = copy.deepcopy(scan_raw)
    scan_raw.set_distances(np.ones(3) * 2.0)
    scan_raw.set_intensities(np.ones(3) * 3.0)
    print(f"scan_raw: {scan_raw}")
    print(f"scan_equal: {scan_equal}")
    print(f"scan_copy: {scan_copy}")
    print(f"scan_deepcopy: {scan_deepcopy}")

    print("\n#### scan_stamped copy ####")
    scan_stamped_equal = None
    scan_stamped_equal = scan_stamped_raw
    scan_stamped_copy = copy.copy(scan_stamped_raw)
    scan_stamped_deepcopy = copy.deepcopy(scan_stamped_raw)
    scan_stamped_raw.stamp().set_time(3)
    scan_stamped_raw.scan().set_distances(np.ones(3) * 2.0)
    scan_stamped_raw.scan().set_intensities(np.ones(3) * 3.0)
    print(f"scan_stamped_raw: {scan_stamped_raw}")
    print(f"scan_stamped_equal: {scan_stamped_equal}")
    print(f"scan_stamped_copy: {scan_stamped_copy}")
    print(f"scan_stamped_deepcopy: {scan_stamped_deepcopy}")


if __name__ == "__main__":
    main()
