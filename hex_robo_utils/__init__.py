#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-01-14
################################################################

# utils
from .dyn_util import DynUtil
from .obs_util import ObsUtilJoint
from .obs_util import ObsUtilWork
from .hex_hdf5_reader import HexHdf5Reader
from .hex_hdf5_writer import HexHdf5Writer

# basic
from .math_utils import hat
from .math_utils import vee
from .math_utils import rad2deg
from .math_utils import deg2rad
from .math_utils import angle_norm
from .math_utils import quat_slerp
from .math_utils import quat_mul
from .math_utils import quat_inv
from .math_utils import trans_inv

# rotation
from .math_utils import rot2quat
from .math_utils import rot2axis
from .math_utils import rot2so3
from .math_utils import quat2rot
from .math_utils import quat2axis
from .math_utils import quat2so3
from .math_utils import axis2rot
from .math_utils import axis2quat
from .math_utils import axis2so3
from .math_utils import so32rot
from .math_utils import so32quat
from .math_utils import so32axis

# pose
from .math_utils import trans2part
from .math_utils import trans2se3
from .math_utils import part2trans
from .math_utils import part2se3
from .math_utils import se32trans
from .math_utils import se32part

# euler
from .math_utils import zyz2rot
from .math_utils import rot2zyz
from .math_utils import yaw2quat
from .math_utils import quat2yaw

__all__ = [
    # version
    '__version__',

    # utils
    'DynUtil',
    'ObsUtilJoint',
    'ObsUtilWork',
    'HexHdf5Reader',
    'HexHdf5Writer',

    # math basic
    'hat',
    'vee',
    'rad2deg',
    'deg2rad',
    'angle_norm',
    'quat_slerp',
    'quat_mul',
    'quat_inv',
    'trans_inv',

    # math rotation
    'rot2quat',
    'rot2axis',
    'rot2so3',
    'quat2rot',
    'quat2axis',
    'quat2so3',
    'axis2rot',
    'axis2quat',
    'axis2so3',
    'so32rot',
    'so32quat',
    'so32axis',

    # math pose
    'trans2part',
    'trans2se3',
    'part2trans',
    'part2se3',
    'se32trans',
    'se32part',

    # math euler
    'zyz2rot',
    'rot2zyz',
    'yaw2quat',
    'quat2yaw',
]

# print("#### Thanks for using HEXFELLOW Utilities :) ####")
