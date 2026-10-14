#!/usr/bin/env python3
# arm_controller_complete.py
"""
完整的机械臂控制器 - 支持圆弧移动、轨迹规划
"""
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState
import numpy as np
import math
import time
from threading import Lock
from typing import Optional, List, Tuple

class ArmControllerWithGripper(Node):
    def __init__(self, node: Node = None):
        if node is None:
            super().__init__('arm_controller')
            self.own_node = True
        else:
            self.node = node
            self.own_node = False
        
        # ✅ 添加所有必需的属性
        self.joint_state_lock = Lock()  # 添加锁
        self.current_joint_positions = [0.0] * 6  # 6个关节的当前位置
        self.latest_joint_state = None
        self.joint_state_received = False
        
        # 关节名称
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
            'finger_joint'
        ]
        
        # 轨迹发布者
        self.trajectory_pub = self.node.create_publisher(
            JointTrajectory,
            '/moveit_trajectory',
            10
        )
        
        # IK服务
        self.ik_client = self.node.create_client(
            GetPositionIK,
            '/compute_ik'
        )
        
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("等待IK服务...")
        
        # ✅ 订阅关节状态话题
        self.joint_state_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # 等待关节状态
        self.node.get_logger().info("等待关节状态消息...")
        start_time = time.time()
        while not self.joint_state_received and time.time() - start_time < 3.0:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            #time.sleep(0.01)
        
        if not self.joint_state_received:
            self.node.get_logger().warn("未收到关节状态，使用默认值")
            self.current_joint_positions = [0.0, -1.0, 1.0, -1.0, -1.57, 0.0]
        
        # 重要：定义固定的夹爪朝向
       # self.fixed_orientation = [0.9260868302943354, -0.3757392780216443, 0.018674115341330632, -0.028886936898587273]
        self.fixed_orientation = [1.0,0.0,0.0,0.0]
        
        # 当前状态
        self.current_position = [0.5, 0.0, 0.4]  # 更合理的初始位置
        self.home_position = [0.5,0.36,0.5]
        self.current_gripper = 0.08
        self.gripper_open = 0.8
        self.gripper_close = 0.0
        self.grasp_force = 50.0
        
 
        # 默认抓取高度
        self.pick_height = 0.25
        self.approach_height = 0.4
        
        self.node.get_logger().info("机械臂控制器已初始化")
    
    def joint_state_callback(self, msg: JointState):
        """关节状态回调函数"""
        with self.joint_state_lock:
            self.latest_joint_state = msg
            
            # 提取关节位置
            positions = []
            for joint_name in self.joint_names[:6]:
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    positions.append(msg.position[idx])
                else:
                    positions.append(0.0)
            
            if len(positions) == 6:
                self.current_joint_positions = positions
                if not self.joint_state_received:
                    self.joint_state_received = True
                    self.node.get_logger().info(f"收到关节状态: {positions}")
    
    def get_current_joint_positions(self) -> List[float]:
        """获取当前关节位置"""
        with self.joint_state_lock:
            return self.current_joint_positions.copy()
    
    def euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> List[float]:
        """欧拉角转四元数"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        return [
            sr * cp * cy - cr * sp * sy,  # x
            cr * sp * cy + sr * cp * sy,  # y
            cr * cp * sy - sr * sp * cy,  # z
            cr * cp * cy + sr * sp * sy   # w
        ]
    
    def get_fixed_orientation(self, orientation_type: str = "down") -> List[float]:
        """获取固定朝向"""
        if orientation_type == "down":
            return self.fixed_orientation
        elif orientation_type == "forward":
            return self.euler_to_quaternion(0, 0, 0)
        elif orientation_type == "up":
            return self.euler_to_quaternion(0, math.pi/2, 0)
        else:
            return self.fixed_orientation
    
    def compute_ik_single(self, position: List[float], orientation: List[float], seed: List[float]) -> Optional[List[float]]:
        """单次IK计算"""
        # 确保seed是6个关节
        if seed is None or len(seed) != 6:
            self.node.get_logger().warn(f"种子长度错误: {len(seed) if seed else 'None'}，使用默认种子")
            seed = [0.0, -1.0, 1.0, -1.0, -1.57, 0.0]
        
        # 创建IK请求
        ik_request = PositionIKRequest()
        
        # 设置机器人状态
        robot_state = RobotState()
        joint_state = JointState()
        joint_state.name = self.joint_names[:6]
        joint_state.position = seed
        robot_state.joint_state = joint_state
        ik_request.robot_state = robot_state
        
        # 设置目标姿态
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = 'world'
        pose_stamped.header.stamp = self.node.get_clock().now().to_msg()
        pose_stamped.pose.position.x = position[0]
        pose_stamped.pose.position.y = position[1]
        pose_stamped.pose.position.z = position[2]
        pose_stamped.pose.orientation.x = orientation[0]
        pose_stamped.pose.orientation.y = orientation[1]
        pose_stamped.pose.orientation.z = orientation[2]
        pose_stamped.pose.orientation.w = orientation[3]
        
        ik_request.pose_stamped = pose_stamped
        ik_request.group_name = 'ur5_arm'
        ik_request.avoid_collisions = True
        ik_request.timeout.sec = 2
        
        # 调用IK服务
        request = GetPositionIK.Request()
        request.ik_request = ik_request
        
        future = self.ik_client.call_async(request)
        
        # 等待响应
        start_time = time.time()
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self.node, timeout_sec=0.1)
            #time.sleep(0.01)
            if time.time() - start_time > 3.0:
                return None
        
        if future.result() is not None:
            response = future.result()
            if response.error_code.val == 1:  # SUCCESS
                return list(response.solution.joint_state.position)[:6]
        
        return None
    
    def move_in_arc_simple(self, start_pos: List[float], end_pos: List[float], radius: float = 0.2,
                          clockwise: bool = True, gripper_position: Optional[float] = None,
                          total_duration: float = 2.0, num_points: int = 8) -> bool:
        """
        简化的圆弧移动：逐点移动，避免复杂的轨迹规划
        """
        if gripper_position is None:
            gripper_position = self.current_gripper
        
        orientation = self.get_fixed_orientation("down")
        
        # 使用起点高度
        plane_z = start_pos[2]
        
        # 计算距离
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        chord_length = np.sqrt(dx*dx + dy*dy)
        
        if chord_length < 0.001:
            self.node.get_logger().info("距离太近，直接移动")
            return self.move_to_position(end_pos, gripper_position, total_duration)
        
        # 自动调整半径
        if radius is None:
            radius = chord_length * 0.7
        
        if radius < chord_length / 2:
            radius = chord_length / 2
        
        # 计算圆心
        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2
        
        if chord_length > 0.001:
            chord_dir_x = dx / chord_length
            chord_dir_y = dy / chord_length
        else:
            chord_dir_x = 0
            chord_dir_y = 0
        
        # 计算h
        h_squared = radius**2 - (chord_length/2)**2
        if h_squared < 0:
            if abs(h_squared) < 0.001:
                h = 0.0
            else:
                self.node.get_logger().error(f"无法计算圆弧，h_squared={h_squared}")
                return self.move_linearly(start_pos, end_pos, 3, gripper_position, total_duration/3)
        else:
            h = np.sqrt(h_squared)
        
        # 计算圆心
        if clockwise:
            center_x = mid_x - chord_dir_y * h
            center_y = mid_y + chord_dir_x * h
        else:
            center_x = mid_x + chord_dir_y * h
            center_y = mid_y - chord_dir_x * h
        
        # 计算角度
        start_angle = np.arctan2(start_pos[1] - center_y, start_pos[0] - center_x)
        end_angle = np.arctan2(end_pos[1] - center_y, end_pos[0] - center_x)
        
        if clockwise:
            if end_angle > start_angle:
                end_angle -= 2 * np.pi
        else:
            if end_angle < start_angle:
                end_angle += 2 * np.pi
        
        # 生成路径点
        waypoints = []
        angles = np.linspace(start_angle, end_angle, num_points + 1)
        
        for angle in angles:
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            waypoints.append([x, y, plane_z])
        
        self.node.get_logger().info(f"圆弧移动: 半径={radius:.3f}m, 点={len(waypoints)}个")
        
        # 逐点移动
        point_duration = total_duration / len(waypoints)
        
        for i, pos in enumerate(waypoints):
            self.node.get_logger().debug(f"圆弧点 {i+1}/{len(waypoints)}")
            
            if not self.move_to_position_direct(pos, gripper_position, point_duration):
                self.node.get_logger().warn(f"点 {i} 移动失败，尝试继续")
                continue
            
            time.sleep(point_duration * 0.1)
        
        return True
    
    def move_to_position_direct(self, position: List[float], gripper_position: Optional[float] = None,
                               duration: float = 1.0, orientation_type: str = "down") -> bool:
        """直接移动到位置"""
        if gripper_position is None:
            gripper_position = self.current_gripper
        
        orientation = self.get_fixed_orientation(orientation_type)
        seed = self.get_current_joint_positions()
        
        solution = self.compute_ik_single(position, orientation, seed)
        if solution is None:
            self.node.get_logger().error(f"IK失败: {position}")
            return False
        
        return self.send_joint_command(solution, gripper_position, duration)
    
    def send_joint_command(self, arm_positions: List[float], gripper_position: float, duration: float = 2.0) -> bool:
        """发送关节命令"""
        joint_positions = list(arm_positions) + [gripper_position]
        
        msg = JointTrajectory()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        
        msg.points = [point]
        self.trajectory_pub.publish(msg)
        
        # 更新状态
        self.current_gripper = gripper_position
        with self.joint_state_lock:
            self.current_joint_positions = list(arm_positions)
        
        self.node.get_logger().info(f"发送关节命令，时间: {duration}s")
        time.sleep(0.1)
        
        return True
    
    def move_to_position(self, position: List[float], gripper_position: Optional[float] = None,
                        duration: float = 2.0, orientation_type: str = "down") -> bool:
        """移动到指定位置"""
        return self.move_to_position_direct(position, gripper_position, duration, orientation_type)
    

        # 修复并简化后的move_linearly方法（核心）
    
    def move_linearly(self, start_pos: List[float], end_pos: List[float], num_points: int = 5,
                    gripper_position: Optional[float] = None, duration_per_step: float = 0.5,  # 恢复旧参数名
                    orientation_type: str = "down", frame_id='world', total_duration: float = None) -> bool:
        """
        线性移动（兼容旧参数名 duration_per_step，解决朝向滞后问题）
        """
        # 兼容逻辑：如果传了duration_per_step但没传total_duration，自动计算总时长
        if total_duration is None:
            # 总时长 = 单步时长 × (插值点数 - 1)（因为起止点之间有num_points-1段）
            total_duration = duration_per_step * (num_points - 1)
        
        if gripper_position is None:
            gripper_position = self.current_gripper

        # 1. 获取固定朝向（全程不变）
        fixed_orientation = self.get_fixed_orientation(orientation_type)
        if fixed_orientation is None:
            self.node.get_logger().error("获取固定朝向失败")
            return False

        # 2. 笛卡尔空间插值生成所有位置点
        cartesian_points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            interpolated_pos = [
                start_pos[0] + t * (end_pos[0] - start_pos[0]),
                start_pos[1] + t * (end_pos[1] - start_pos[1]),
                start_pos[2] + t * (end_pos[2] - start_pos[2])
            ]
            cartesian_points.append(interpolated_pos)

        # 3. 为每个位置点求解IK（统一朝向约束）
        joint_waypoints = []
        last_joints = self.get_current_joint_positions()
        for idx, pos in enumerate(cartesian_points):
            current_joints = self.compute_ik_single(pos, fixed_orientation, last_joints)
            if current_joints is None:
                self.node.get_logger().error(f"路径点 {idx+1} IK求解失败: {pos}")
                return False
            joint_waypoints.append(current_joints)
            last_joints = current_joints

        # 4. 构建完整的关节轨迹
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = self.node.get_clock().now().to_msg()
        traj_msg.header.frame_id = frame_id  # 兼容传入的frame_id
        traj_msg.joint_names = self.joint_names

        # 5. 填充轨迹点（按总时长分配）
        time_per_point = total_duration / (num_points - 1) if num_points > 1 else total_duration
        for idx, arm_joints in enumerate(joint_waypoints):
            point = JointTrajectoryPoint()
            point.positions = list(arm_joints) + [gripper_position]
            total_time = idx * time_per_point
            point.time_from_start.sec = int(total_time)
            point.time_from_start.nanosec = int((total_time - int(total_time)) * 1e9)
            traj_msg.points.append(point)

        # 6. 发送完整轨迹
        self.trajectory_pub.publish(traj_msg)
        self.node.get_logger().info(f"发送完整轨迹：{num_points}个点，总时长{total_duration}s（单步{duration_per_step}s）")

        # 7. 更新状态 + 等待执行
        self.current_gripper = gripper_position
        with self.joint_state_lock:
            self.current_joint_positions = joint_waypoints[-1]
        self.current_position = end_pos

        time.sleep(total_duration + 0.5)
        return True

    # # 以下是你已有的方

    # def move_linearly(self, start_pos: List[float], end_pos: List[float], num_points: int = 5,
    #                  gripper_position: Optional[float] = None, duration_per_step: float = 0.5) -> bool:
    #     """线性移动"""
    #     if gripper_position is None:
    #         gripper_position = self.current_gripper
        
    #     # 生成路径
    #     positions = []
    #     for i in range(num_points + 1):
    #         t = i / num_points
    #         pos = [
    #             start_pos[0] + t * (end_pos[0] - start_pos[0]),
    #             start_pos[1] + t * (end_pos[1] - start_pos[1]),
    #             start_pos[2] + t * (end_pos[2] - start_pos[2])
    #         ]
    #         positions.append(pos)
        
    #     # 移动到每个点
    #     for i, pos in enumerate(positions):
    #         self.node.get_logger().info(f"路径点 {i+1}/{len(positions)}: {pos}")
    #         if not self.move_to_position_direct(pos, gripper_position, duration_per_step):
    #             return False
        
    #     return True
    
    def go_home(self, gripper_position: Optional[float] = None) -> bool:
        """回到初始位置"""
        if gripper_position is None:
            gripper_position = self.gripper_open
        

        
        self.node.get_logger().info("回到初始位置...")
        success = self.move_to_position_direct(self.home_position, gripper_position, 1.5)
        
        if success:
            self.node.get_logger().info("✅ 已回到初始位置")
        else:
            self.node.get_logger().error("❌ 回到初始位置失败")
        
        return success
    
    def control_gripper(self, open: bool = True, width: Optional[float] = None, duration: float = 1.0) -> bool:
        """控制夹爪"""
        if width is None:
            width = self.gripper_open if open else self.gripper_close
        
        gripper_pos = 0.8 - width
        
        # 保持当前位置，只改变夹爪
        return self.send_joint_command(
            self.get_current_joint_positions(),
            gripper_position=gripper_pos,
            duration=duration
        )
    
    def execute_pick_place(self, object_position: List[float], place_position: Optional[List[float]] = None,
                               object_type: str = "unknown", object_size: float = 0.03) -> bool:
        """
        安全的抓取放置实现
        """
        if place_position is None:
            place_position = [0.3, 0.2, 0.1]
        
        self.node.get_logger().info("开始抓取放置操作")
        
        try:
            # 0. 确保在初始位置
            if not self.go_home():
                return False
            
            # 1. 打开夹爪
            self.node.get_logger().info("1. 打开夹爪")
            self.control_gripper(open=True, width=0.8)
            time.sleep(1)
            
            # 2. 移动到物体上方
            above_pos = [object_position[0], object_position[1],
                        object_position[2] + self.approach_height]
            
            self.node.get_logger().info(f"2. 移动到物体上方: {above_pos}")
            if not self.move_to_position_direct(above_pos, duration=1.5):
                return False
            time.sleep(2)
            
            # 3. 下降到抓取位置
            grasp_pos = [object_position[0], object_position[1],
                        object_position[2] + self.pick_height]
            
            self.node.get_logger().info(f"3. 下降到抓取位置: {grasp_pos}")
            if not self.move_linearly(above_pos, grasp_pos, 5, duration_per_step=0.3):
                return False
            time.sleep(2)
            
            # 4. 抓取
            grasp_width = self._get_grasp_width(object_type, object_size)
            self.node.get_logger().info(f"4. 关闭夹爪 (宽度={grasp_width:.3f}m)")
            self.control_gripper(open=False, width=grasp_width, duration=1.0)
            time.sleep(2)
            
            # 5. 提升
            self.node.get_logger().info("5. 提升物体")
            if not self.move_linearly(grasp_pos, above_pos, 3, duration_per_step=0.2):
                self.control_gripper(open=True)
                return False
            time.sleep(1)

            self.go_home()
            time.sleep(1)
            
            # # 6. 提升到安全高度
            safe_height = 0.5
            safe_pos = [object_position[0], object_position[1], safe_height]
            
            # self.node.get_logger().info(f"6. 提升到安全高度: {safe_height}m")
            # if not self.move_to_position_direct(safe_pos, duration=1.0):
            #     self.control_gripper(open=True)
            #     return False
            # time.sleep(1)
            
            # 7. 圆弧移动到放置位置
            place_safe = [place_position[0], place_position[1], safe_height]
            
            self.node.get_logger().info("7. 圆弧移动到放置位置")
            # if not self.move_to_position_direct(place_safe, duration=2.0):
            #     self.control_gripper(open=True)
            #     return False
            # time.sleep(3)
            try:
                success = self.move_in_arc_simple(
                    start_pos=self.home_position,
                    end_pos=place_safe,
                    radius=0.15,
                    clockwise=True,
                    total_duration=5.0
                )
                
                if not success:
                    self.node.get_logger().warn("圆弧移动失败，使用直线移动")
                    if not self.move_linearly(safe_pos, place_safe, 5, duration_per_step=0.5):
                        self.control_gripper(open=True)
                        return False
            except Exception as e:
                self.node.get_logger().error(f"圆弧移动异常: {e}")
                self.node.get_logger().info("回退到直线移动")
                if not self.move_linearly(safe_pos, place_safe, 5, duration_per_step=0.5):
                    self.control_gripper(open=True)
                    return False
            
            time.sleep(1)
            
            # 8. 放置
            place_above = [place_position[0], place_position[1],
                          place_position[2] + self.approach_height]
            
            self.node.get_logger().info(f"8. 下降到放置位置: {place_above[2]:.3f}m")
            if not self.move_to_position_direct(place_above, duration=1.0):
                self.control_gripper(open=True)
                return False
            time.sleep(1)
            
            place_done = [place_position[0], place_position[1],
                         place_position[2] + self.pick_height]
            
            if not self.move_to_position_direct(place_done, duration=0.8):
                self.control_gripper(open=True)
                return False
            time.sleep(1)
            
            # 9. 释放
            self.node.get_logger().info("9. 放置物体")
            self.control_gripper(open=True, width=0.8, duration=1.0)
            time.sleep(1)
            
            # 10. 提升离开
            self.node.get_logger().info("10. 提升离开")
            if not self.move_to_position_direct(place_safe, duration=0.8):
                self.node.get_logger().warn("提升离开失败")
            time.sleep(1)
            
            # 11. 回到初始位置
            self.node.get_logger().info("11. 回到初始位置")
            self.go_home()
            
            self.node.get_logger().info("✅ 抓取放置完成")
            return True
            
        except Exception as e:
            self.node.get_logger().error(f"抓取放置失败: {str(e)}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())
            
            # 安全措施
            try:
                self.control_gripper(open=True)
                self.go_home()
            except:
                pass
            
            return False
    
    def _get_grasp_width(self, object_type: str, object_size: float) -> float:
        """计算抓取宽度"""
        if object_type in ["bottle", "cup", "cylinder"]:
            width = object_size * 0.9
        elif object_type in ["box", "cube"]:
            width = object_size
        elif object_type in ["sphere", "ball"]:
            width = object_size * 1.1
        else:
            width = object_size * 0.9
        
        width = max(0.01, min(width, self.gripper_open))
        return width


    def get_status(self):
        """获取状态"""
        return {
            'position': self.current_joint_positions,
            'gripper': self.current_gripper,
            'orientation': 'down',  # 固定朝向
            'state': 'home' if self.current_position == [0.3, 0.0, 0.4] else 'working'
        }