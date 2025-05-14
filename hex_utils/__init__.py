#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-01-14
################################################################

__version__ = "0.1.1"

from ._hex_stamp import HexStamp
from ._hex_arm_ctrl import HexArmCtrl, HexArmCtrlStamped
from ._hex_arm_state import HexArmState, HexArmStateStamped
from ._hex_cart_pose import HexCartPose, HexCartPoseStamped
from ._hex_cart_state import HexCartState, HexCartStateStamped
from ._hex_cart_vel import HexCartVel, HexCartVelStamped
from ._hex_sensor_cloud import HexSensorCloudStamped
from ._hex_sensor_imu import HexSensorImu, HexSensorImuStamped
from ._hex_sensor_scan import HexSensorScan, HexSensorScanStamped

from ._math_util import cross_matrix
from ._math_util import rad2deg
from ._math_util import deg2rad
from ._math_util import angle_norm
from ._math_util import quat_slerp
from ._math_util import quat_mul
from ._math_util import quat_inv
from ._math_util import trans_inv
from ._math_util import quat2rot
from ._math_util import rot2quat
from ._math_util import axis2rot
from ._math_util import rot2axis
from ._math_util import quat2axis
from ._math_util import axis2quat
from ._math_util import part2trans
from ._math_util import trans2part
from ._math_util import se32trans
from ._math_util import trans2se3
from ._math_util import zyz2rot
from ._math_util import rot2zyz
from ._math_util import yaw2quat
from ._math_util import quat2yaw

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
    'HexSensorImu',
    'HexSensorImuStamped',
    'HexSensorScan',
    'HexSensorScanStamped',
    # math_util
    'cross_matrix',
    'rad2deg',
    'deg2rad',
    'angle_norm',
    'quat_slerp',
    'quat_mul',
    'quat_inv',
    'trans_inv',
    'quat2rot',
    'rot2quat',
    'axis2rot',
    'rot2axis',
    'quat2axis',
    'axis2quat',
    'part2trans',
    'trans2part',
    'se32trans',
    'trans2se3',
    'zyz2rot',
    'rot2zyz',
    'yaw2quat',
    'quat2yaw',
]

print("#### Thanks for using HEXFELLOW Utilities :) ####")
