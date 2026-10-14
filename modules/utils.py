# modules/utils.py
"""几何工具:四元数/欧拉角/旋转矩阵/齐次变换"""
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def quaternion_to_euler(x, y, z, w):
    """四元数 (x,y,z,w) → 欧拉角 (roll, pitch, yaw)"""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quaternion_to_matrix(quat):
    """四元数 [x,y,z,w] → 旋转矩阵(scipy)"""
    return R.from_quat(quat).as_matrix()


def build_transform(pos, rot, is_quat=True):
    """
    构建齐次变换矩阵
    is_quat=True 时 rot 为 (w,x,y,z);False 时为欧拉角(度)
    """
    T = np.eye(4)
    T[:3, 3] = pos

    if is_quat:
        r = R.from_quat([rot[1], rot[2], rot[3], rot[0]])  # 转为 x,y,z,w
    else:
        r = R.from_euler('xyz', rot, degrees=True)

    T[:3, :3] = r.as_matrix()
    return T


def build_transform_from_matrix(pos, rot_matrix):
    """平移向量 + 旋转矩阵 → 齐次变换矩阵"""
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = rot_matrix
    return T
