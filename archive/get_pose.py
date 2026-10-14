import cv2
import numpy as np

def get_object_pose(rgb, depth, mask,camera_matrix,):
    """
    输入: RGB图, 深度图, 相机内参
    输出: 物体在桌面坐标系下的(x, y, theta)
    """
    # 1. 物体检测（简单阈值分割或颜色分割）

    
    # 2. 计算质心像素坐标
    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    # 3. 从深度图获取物体距离
    depth_value = depth[cy, cx]  # 物体中心深度
    
    # 4. 相机坐标 -> 世界坐标（z=0平面）
    # 相机坐标系下的3D点
    Z = depth_value
    X = (cx - camera_matrix[0,2]) * Z / camera_matrix[0,0]
    Y = (cy - camera_matrix[1,2]) * Z / camera_matrix[1,1]
    
    # 5. 转换到桌面坐标系
    # 假设相机与地面有固定高度H，倾角theta_cam
    x_world, y_world = camera_to_table(X, Y, Z, H, theta_cam)
    
    # 6. 计算朝向（简单版本：主成分分析）
    points = np.column_stack(np.where(mask > 0))
    if len(points) > 2:
        pca = PCA(n_components=2)
        pca.fit(points)
        angle = np.arctan2(pca.components_[0,1], pca.components_[0,0])
    
    return (x_world, y_world, angle)