"""Solve an end-effector observation pose from a desired optical-camera pose.

Transforms use column vectors: T_world_camera = T_world_ee @ T_ee_camera.
The camera frame is ROS optical (+z forward, +x right, +y down).
"""
import numpy as np
from scipy.spatial.transform import Rotation


def camera_centered_pose(target_world, camera_clearance, ee_from_camera, yaw=0.):
    target = np.asarray(target_world, dtype=float)
    mount = np.asarray(ee_from_camera, dtype=float)
    if target.shape != (3,) or mount.shape != (4,4) or not np.isfinite(target).all() or not np.isfinite(mount).all():
        raise ValueError('Invalid camera observation geometry')
    if not np.isfinite(camera_clearance) or camera_clearance <= 0. or not np.isfinite(yaw):
        raise ValueError('Camera clearance must be positive and yaw finite')
    if not np.allclose(mount[3], [0,0,0,1]) or not np.allclose(mount[:3,:3].T@mount[:3,:3], np.eye(3),atol=1e-5):
        raise ValueError('Camera mounting transform must be rigid')
    if not np.isclose(np.linalg.det(mount[:3,:3]),1.,atol=1e-5):
        raise ValueError('Camera mounting rotation must be right-handed')
    # Remove calibration rounding noise before composing rigid transforms.
    mount=mount.copy()
    mount[:3,:3]=Rotation.from_matrix(mount[:3,:3]).as_matrix()
    world_from_camera=np.eye(4)
    world_from_camera[:3,3]=target+[0.,0.,camera_clearance]
    world_from_camera[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()@np.diag([1.,-1.,-1.])
    world_from_ee=world_from_camera@np.linalg.inv(mount)
    return world_from_ee,world_from_camera
