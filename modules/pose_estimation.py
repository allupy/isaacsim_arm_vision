# modules/pose_estimation.py
"""物体 6D 位姿估计:掩码质心+深度 → 基座系位置;掩码 PCA 短轴 → 抓取偏航角

朝向约定:取掩码主轴中**较短**的那根(即方块长边的垂线方向)作为夹爪闭合方向,
使夹爪手指与方块长边垂直。返回四元数 [x,y,z,w]。
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from modules.utils import build_transform_from_matrix


def estimate_object_pose(rgb, depth, ee_in_base, mask, camera_matrix,
                         R_cam2gripper, t_cam2gripper, yaw_offset_deg: float = 0.0):
    """
    参数:
        rgb: HWC BGR 图像(未使用,保持接口一致)
        depth: HW 深度图(米)
        ee_in_base: dict, 含 'rotation_matrix', 'position'(TF 获取)
        mask: 二值掩码(SAM)
        camera_matrix: 3x3 内参
        R_cam2gripper / t_cam2gripper: 手眼标定(相机装在末端)
        yaw_offset_deg: 抓取朝向补偿角(夹爪闭合轴与末端系 x/y 的对齐差,按需 0 或 90)

    返回:
        (x, y, z, quat): 基座标系下位置与抓取朝向四元数 [x,y,z,w]
    """
    # ---- 位置:质心 + 深度 → 相机系 → 基座系 ----
    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    Z = float(depth[cy, cx])

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx_cam = camera_matrix[0, 2]
    cy_cam = camera_matrix[1, 2]

    X_cam = (cx - cx_cam) * Z / fx
    Y_cam = (cy - cy_cam) * Z / fy
    point_camera = np.array([X_cam, Y_cam, Z, 1.0]).reshape(4, 1)

    R_ee2base = ee_in_base['rotation_matrix']
    t_ee2base = np.array(ee_in_base['position']).flatten()
    T_ee2base = build_transform_from_matrix(t_ee2base, R_ee2base)

    T_cam2ee = build_transform_from_matrix(np.asarray(t_cam2gripper).flatten(),
                                           np.asarray(R_cam2gripper))
    T_cam2base = T_ee2base @ T_cam2ee
    point_base = (T_cam2base @ point_camera)[:3].flatten()

    # ---- 朝向:掩码 PCA 短轴方向 → 基座系水平偏航 ----
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) < 10:
        world_angle = 0.0
    else:
        points_pixel = np.column_stack((x_coords, y_coords))
        points_center = points_pixel - np.array([cx, cy])
        cov = np.cov(points_center.T)
        eig_vals, eig_vecs = np.linalg.eig(cov)
        short_vec_pixel = eig_vecs[:, np.argsort(eig_vals)[0]]  # 短轴

        u, v = np.real(short_vec_pixel)
        dir_cam = np.array([u / fx, v / fy, 1.0])
        dir_cam = dir_cam / np.linalg.norm(dir_cam)

        dir_base = T_cam2base[:3, :3] @ dir_cam
        dir_xy = np.real(dir_base[:2])
        if np.linalg.norm(dir_xy) < 1e-6:
            world_angle = 0.0
        else:
            dir_xy = dir_xy / np.linalg.norm(dir_xy)
            world_angle = np.degrees(np.arctan2(dir_xy[1], dir_xy[0]))

    world_angle = (world_angle + yaw_offset_deg) % 180.0

    # 基准朝向(与夹爪固连的默认朝向)绕 z 轴旋转 world_angle
    fixed_rot = R.from_quat([1.0, 0.0, 0.0, 0.0]).as_matrix()
    z_rot = R.from_euler('z', world_angle, degrees=True).as_matrix()
    final_quat = R.from_matrix(z_rot @ fixed_rot).as_quat()  # [x,y,z,w]

    return point_base[0], point_base[1], point_base[2], final_quat
