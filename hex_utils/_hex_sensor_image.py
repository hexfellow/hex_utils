#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-05-26
################################################################

import copy
import numpy as np

from hex_utils._hex_stamp import HexStamp


class HexSensorImageStamped:

    def __init__(
            self,
            stamp=HexStamp(),
            image=np.zeros((640, 480, 3), dtype=np.uint8),
    ):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set stamp type err: {type(stamp)}")
        if self.__is_image(image):
            self.__image = copy.deepcopy(image)
        else:
            raise TypeError(f"set image type err: {type(image)}")

    def __repr__(self):
        print_str = f"stamp: {self.__stamp}\n"
        print_str += f"image: {self.__image.shape}"
        return print_str

    def __is_stamp(self, stamp):
        return isinstance(stamp, HexStamp)

    def __is_image(self, image):
        return isinstance(image, np.ndarray)

    def stamp(self):
        return self.__stamp

    def image(self):
        return self.__image

    def get_stamp(self):
        return copy.deepcopy(self.__stamp)

    def get_image(self):
        return copy.deepcopy(self.__image)

    def set_stamp(self, stamp):
        if self.__is_stamp(stamp):
            self.__stamp = copy.deepcopy(stamp)
        else:
            raise TypeError(f"set stamp type err: {type(stamp)}")

    def set_image(self, image):
        if self.__is_image(image):
            self.__image = copy.deepcopy(image)
        else:
            raise TypeError(f"set image type err: {type(image)}")


def main():
    stamp = HexStamp(1, 2)
    image_raw = np.ones((640, 480, 3), dtype=np.uint8) * 127
    image_stamped_raw = HexSensorImageStamped(stamp, image_raw)
    print(f"image_stamped_raw: {image_stamped_raw}")

    print("\n#### image copy ####")
    image_equal = None
    image_equal = image_raw
    image_copy = copy.copy(image_raw)
    image_deepcopy = copy.deepcopy(image_raw)
    image_raw = np.ones((320, 240, 3), dtype=np.uint8) * 255
    print(f"image_raw: {image_raw.shape}")
    print(f"image_equal: {image_equal.shape}")
    print(f"image_copy: {image_copy.shape}")
    print(f"image_deepcopy: {image_deepcopy.shape}")

    print("\n#### image_stamped copy ####")
    image_stamped_equal = None
    image_stamped_equal = image_stamped_raw
    image_stamped_copy = copy.copy(image_stamped_raw)
    image_stamped_deepcopy = copy.deepcopy(image_stamped_raw)
    image_stamped_raw.stamp().set_time(3)
    image_stamped_raw.set_image(np.ones((320, 240, 3), dtype=np.uint8) * 255)
    print(f"image_stamped_raw: {image_stamped_raw}")
    print(f"image_stamped_equal: {image_stamped_equal}")
    print(f"image_stamped_copy: {image_stamped_copy}")
    print(f"image_stamped_deepcopy: {image_stamped_deepcopy}")


if __name__ == "__main__":
    main()
