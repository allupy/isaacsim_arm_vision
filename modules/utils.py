import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf2_ros
import time
import os
import math
import json
from datetime import datetime


def quaternion_to_euler(x, y, z, w):
    """四元数转欧拉角"""
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

from scipy.spatial.transform import Rotation as R

def quaternion_to_matrix(quat):
    """
    使用scipy转换四元数为旋转矩阵
    四元数格式: [x, y, z, w]
    """
    rotation = R.from_quat(quat)
    return rotation.as_matrix()


def build_transform(pos, rot, is_quat=True):
    """
    构建齐次变换矩阵
    rot: w,x,y,z
    """
    T = np.eye(4)
    T[:3, 3] = pos
    
    if is_quat:
        r = R.from_quat([rot[1], rot[2], rot[3],rot[0]]) # 变换为x,y,z,w
    else:
        # 欧拉角转旋转矩阵
        r = R.from_euler('xyz', rot, degrees=True)
    
    T[:3, :3] = r.as_matrix()
    return T

def build_transform_from_matrix(pos, rot_matrix):
    """
    从旋转矩阵和平移向量构建变换矩阵
    """
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = rot_matrix
    return T