#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-01-14
################################################################

__version__ = "0.1.7"

from ._hex_stamp import HexStamp
from ._hex_arm_ctrl import HexArmCtrl, HexArmCtrlStamped
from ._hex_arm_state import HexArmState, HexArmStateStamped
from ._hex_cart_pose import HexCartPose, HexCartPoseStamped
from ._hex_cart_state import HexCartState, HexCartStateStamped
from ._hex_cart_vel import HexCartVel, HexCartVelStamped
from ._hex_sensor_cloud import HexSensorCloudStamped
from ._hex_sensor_image import HexSensorImageStamped
from ._hex_sensor_imu import HexSensorImu, HexSensorImuStamped
from ._hex_sensor_imu import HexSensorImuQuat, HexSensorImuQuatStamped
from ._hex_sensor_mag import HexSensorMag, HexSensorMagStamped
from ._hex_sensor_scan import HexSensorScan, HexSensorScanStamped

# utils
from ._dyn_util import DynUtil
from ._obs_util import ObsUtilJoint
from ._obs_util import ObsUtilWork

# basic
from ._math_utils import hat
from ._math_utils import vee
from ._math_utils import rad2deg
from ._math_utils import deg2rad
from ._math_utils import angle_norm
from ._math_utils import quat_slerp
from ._math_utils import quat_mul
from ._math_utils import quat_inv
from ._math_utils import trans_inv

# rotation
from ._math_utils import rot2quat
from ._math_utils import rot2axis
from ._math_utils import rot2so3
from ._math_utils import quat2rot
from ._math_utils import quat2axis
from ._math_utils import quat2so3
from ._math_utils import axis2rot
from ._math_utils import axis2quat
from ._math_utils import axis2so3
from ._math_utils import so32rot
from ._math_utils import so32quat
from ._math_utils import so32axis

# pose
from ._math_utils import trans2part
from ._math_utils import trans2se3
from ._math_utils import part2trans
from ._math_utils import part2se3
from ._math_utils import se32trans
from ._math_utils import se32part

# euler
from ._math_utils import zyz2rot
from ._math_utils import rot2zyz
from ._math_utils import yaw2quat
from ._math_utils import quat2yaw

__all__ = [
    # version
    '__version__',
    # hex_struct
    'HexStamp',
    'HexArmCtrl',
    'HexArmCtrlStamped',
    'HexArmState',
    'HexArmStateStamped',
    'HexCartVel',
    'HexCartVelStamped',
    'HexCartPose',
    'HexCartPoseStamped',
    'HexCartState',
    'HexCartStateStamped',
    'HexSensorCloudStamped',
    'HexSensorImageStamped',
    'HexSensorImu',
    'HexSensorImuStamped',
    'HexSensorImuQuat',
    'HexSensorImuQuatStamped',
    'HexSensorMag',
    'HexSensorMagStamped',
    'HexSensorScan',
    'HexSensorScanStamped',
    # utils
    'DynUtil',
    'ObsUtilJoint',
    'ObsUtilWork',
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

print("#### Thanks for using HEXFELLOW Utilities :) ####")
