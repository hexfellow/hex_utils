#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-04-21
################################################################

import copy
import open3d as o3d
import open3d.core as o3c

from hex_utils._hex_stamp import HexStamp


class HexSensorCloudStamped:

    def __init__(
            self,
            stamp=HexStamp(),
            cloud=o3d.t.geometry.PointCloud(),
    ):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set stamp type err: {type(stamp)}")
        if self.__is_cloud(cloud):
            self.__cloud = copy.deepcopy(cloud)
        else:
            raise TypeError(f"set cloud type err: {type(cloud)}")

    def __repr__(self):
        print_str = f"stamp: {self.__stamp}\n"
        print_str += f"cloud: {self.__cloud}"
        return print_str

    def __is_stamp(self, stamp):
        return isinstance(stamp, HexStamp)

    def __is_cloud(self, cloud):
        return isinstance(cloud, o3d.t.geometry.PointCloud)

    def stamp(self):
        return self.__stamp

    def cloud(self):
        return self.__cloud

    def get_stamp(self):
        return copy.deepcopy(self.__stamp)

    def get_cloud(self):
        return copy.deepcopy(self.__cloud)

    def set_stamp(self, stamp):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set stamp type err: {type(stamp)}")

    def set_cloud(self, cloud):
        if self.__is_cloud(cloud):
            self.__cloud = copy.deepcopy(cloud)
        else:
            raise TypeError(f"set cloud type err: {type(cloud)}")


def main():
    stamp = HexStamp(1, 2)
    cloud_raw = o3d.t.geometry.PointCloud(
        o3c.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=o3c.float32))
    cloud_raw.point.colors = o3c.Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                                        dtype=o3c.float32)
    cloud_raw.point.normals = o3c.Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                                         dtype=o3c.float32)
    cloud_raw.point.labels = o3c.Tensor([1, 2], dtype=o3c.int32)
    cloud_stamped_raw = HexSensorCloudStamped(stamp, cloud_raw)
    print(f"cloud_stamped_raw: {cloud_stamped_raw}")

    print("\n#### cloud copy ####")
    cloud_equal = None
    cloud_equal = cloud_raw
    cloud_copy = copy.copy(cloud_raw)
    cloud_deepcopy = copy.deepcopy(cloud_raw)
    cloud_raw.point.positions = o3c.Tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=o3c.float32)
    cloud_raw.point.colors = o3c.Tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=o3c.float32)
    cloud_raw.point.normals = o3c.Tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=o3c.float32)
    cloud_raw.point.labels = o3c.Tensor([1, 2, 3], dtype=o3c.int32)
    print(f"cloud_raw: {cloud_raw}")
    print(f"cloud_equal: {cloud_equal}")
    print(f"cloud_copy: {cloud_copy}")
    print(f"cloud_deepcopy: {cloud_deepcopy}")


if __name__ == "__main__":
    main()
