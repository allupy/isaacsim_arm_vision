#!/usr/bin/env python3
"""
机械臂控制器:IK 求解 + 关节轨迹发布,支持圆弧移动与直线插值移动

依赖外部节点(主程序的 ROS 节点)提供:
  - /compute_ik 服务(MoveIt)
  - /joint_states 订阅(当前关节角)
  - /moveit_trajectory 发布(轨迹命令,经转换器转发给 Isaac Sim)

夹爪语义:finger_joint 取值范围 [0, 0.8],0 = 全开,0.8 = 闭合。
本类的 gripper "width" 参数按 `关节值 = 0.8 - width` 映射,
即 width=0.8 → 关节 0(全开),width≈0 → 关节 0.8(夹紧)。
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
from typing import Optional, List

# 默认参数(可被 config/default_config.yaml 的 arm: 段覆盖)
ARM_DEFAULTS = {
    'home_position': [0.3, 0.3, 0.5],   # 初始/安全位
    'pick_height': 0.25,                # 抓取时末端相对物体目标点的抬高
    'approach_height': 0.4,             # 接近阶段末端相对物体目标点的抬高
    'safe_height': 0.5,                 # 长距离移动的过渡高度
    'arc_radius': 0.2,                  # 圆弧移动半径
    'gripper_open': 0.8,                # 夹爪宽度参数(见文件头说明)
    'gripper_close': 0.0,
    'orientation': [1.0, 0.0, 0.0, 0.0],  # 末端固定朝向(x,y,z,w)
    'default_place_position': [0.3, 0.2, 0.1],
}


class ArmControllerWithGripper:
    """UR5 + 夹爪控制器(依赖传入的 ROS 节点)"""

    def __init__(self, node: Node, params: Optional[dict] = None):
        self.node = node

        # 参数(默认值 ← config 覆盖)
        cfg = dict(ARM_DEFAULTS)
        if params:
            cfg.update({k: v for k, v in params.items() if k in cfg})
        self.home_position = list(cfg['home_position'])
        self.pick_height = cfg['pick_height']
        self.approach_height = cfg['approach_height']
        self.safe_height = cfg['safe_height']
        self.arc_radius = cfg['arc_radius']
        self.gripper_open = cfg['gripper_open']
        self.gripper_close = cfg['gripper_close']
        self.fixed_orientation = list(cfg['orientation'])
        self.default_place_position = list(cfg['default_place_position'])

        # 关节状态
        self.joint_state_lock = Lock()
        self.current_joint_positions = [0.0] * 6
        self.latest_joint_state = None
        self.joint_state_received = False

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
            'finger_joint'
        ]

        # IK 求解失败时的默认种子
        self.default_seed = [0.0, -1.0, 1.0, -1.0, -1.57, 0.0]

        # 轨迹发布者
        self.trajectory_pub = self.node.create_publisher(
            JointTrajectory, '/moveit_trajectory', 10)

        # IK 服务
        self.ik_client = self.node.create_client(GetPositionIK, '/compute_ik')
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("等待IK服务...")

        # 关节状态订阅
        self.joint_state_sub = self.node.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        # 等待首帧关节状态(最多3秒)
        self.node.get_logger().info("等待关节状态消息...")
        start_time = time.time()
        while not self.joint_state_received and time.time() - start_time < 3.0:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        if not self.joint_state_received:
            self.node.get_logger().warn("未收到关节状态，使用默认值")
            self.current_joint_positions = list(self.default_seed)

        # 当前状态(估计值,仅用于显示)
        self.current_position = [0.5, 0.0, 0.4]
        self.current_gripper = 0.08

        self.node.get_logger().info("机械臂控制器已初始化")

    # ------------------------------------------------------------------
    # 关节状态
    # ------------------------------------------------------------------
    def joint_state_callback(self, msg: JointState):
        with self.joint_state_lock:
            self.latest_joint_state = msg
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
        with self.joint_state_lock:
            return self.current_joint_positions.copy()

    # ------------------------------------------------------------------
    # 朝向
    # ------------------------------------------------------------------
    def euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> List[float]:
        """欧拉角 → 四元数 (x,y,z,w)"""
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        return [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy
        ]

    def get_fixed_orientation(self, orientation_type: str = "down") -> List[float]:
        if orientation_type == "down":
            return self.fixed_orientation
        elif orientation_type == "forward":
            return self.euler_to_quaternion(0, 0, 0)
        elif orientation_type == "up":
            return self.euler_to_quaternion(0, math.pi / 2, 0)
        return self.fixed_orientation

    # ------------------------------------------------------------------
    # IK 与轨迹发送
    # ------------------------------------------------------------------
    def compute_ik_single(self, position: List[float], orientation: List[float],
                          seed: List[float]) -> Optional[List[float]]:
        """单次IK求解,失败返回None"""
        if seed is None or len(seed) != 6:
            self.node.get_logger().warn("IK种子长度错误，使用默认种子")
            seed = list(self.default_seed)

        ik_request = PositionIKRequest()
        robot_state = RobotState()
        joint_state = JointState()
        joint_state.name = self.joint_names[:6]
        joint_state.position = seed
        robot_state.joint_state = joint_state
        ik_request.robot_state = robot_state

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

        request = GetPositionIK.Request()
        request.ik_request = ik_request

        future = self.ik_client.call_async(request)
        start_time = time.time()
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start_time > 3.0:
                return None

        if future.result() is not None:
            response = future.result()
            if response.error_code.val == 1:  # SUCCESS
                return list(response.solution.joint_state.position)[:6]
        return None

    def send_joint_command(self, arm_positions: List[float], gripper_position: float,
                           duration: float = 2.0) -> bool:
        """发送单点关节命令"""
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

        # 更新本地估计
        self.current_gripper = gripper_position
        with self.joint_state_lock:
            self.current_joint_positions = list(arm_positions)

        time.sleep(0.1)
        return True

    # ------------------------------------------------------------------
    # 运动原语
    # ------------------------------------------------------------------
    def move_to_position_direct(self, position: List[float], gripper_position: Optional[float] = None,
                                duration: float = 1.0, orientation_type: str = "down") -> bool:
        """IK 求解后直接移动到位置(单点轨迹)"""
        if gripper_position is None:
            gripper_position = self.current_gripper

        orientation = self.get_fixed_orientation(orientation_type)
        seed = self.get_current_joint_positions()

        solution = self.compute_ik_single(position, orientation, seed)
        if solution is None:
            self.node.get_logger().error(f"IK失败: {position}")
            return False
        return self.send_joint_command(solution, gripper_position, duration)

    def move_to_position(self, position: List[float], gripper_position: Optional[float] = None,
                         duration: float = 2.0, orientation_type: str = "down") -> bool:
        """移动到指定位置(move_to_position_direct 的别名,保持旧接口)"""
        return self.move_to_position_direct(position, gripper_position, duration, orientation_type)

    def move_linearly(self, start_pos: List[float], end_pos: List[float], num_points: int = 5,
                      gripper_position: Optional[float] = None, duration_per_step: float = 0.5,
                      orientation_type: str = "down", frame_id: str = 'world',
                      total_duration: Optional[float] = None) -> bool:
        """
        笛卡尔空间线性插值移动:整条轨迹一次发出,朝向全程固定。
        兼容旧参数 duration_per_step;传入 total_duration 则以其为准。
        """
        if total_duration is None:
            total_duration = duration_per_step * (num_points - 1)

        if gripper_position is None:
            gripper_position = self.current_gripper

        fixed_orientation = self.get_fixed_orientation(orientation_type)

        # 1. 笛卡尔插值
        cartesian_points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            cartesian_points.append([
                start_pos[0] + t * (end_pos[0] - start_pos[0]),
                start_pos[1] + t * (end_pos[1] - start_pos[1]),
                start_pos[2] + t * (end_pos[2] - start_pos[2]),
            ])

        # 2. 逐点IK(以前一解为种子,保证解的连续性)
        joint_waypoints = []
        last_joints = self.get_current_joint_positions()
        for idx, pos in enumerate(cartesian_points):
            current_joints = self.compute_ik_single(pos, fixed_orientation, last_joints)
            if current_joints is None:
                self.node.get_logger().error(f"路径点 {idx + 1} IK求解失败: {pos}")
                return False
            joint_waypoints.append(current_joints)
            last_joints = current_joints

        # 3. 组装整条轨迹并发布
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = self.node.get_clock().now().to_msg()
        traj_msg.header.frame_id = frame_id
        traj_msg.joint_names = self.joint_names

        time_per_point = total_duration / (num_points - 1) if num_points > 1 else total_duration
        for idx, arm_joints in enumerate(joint_waypoints):
            point = JointTrajectoryPoint()
            point.positions = list(arm_joints) + [gripper_position]
            total_time = idx * time_per_point
            point.time_from_start.sec = int(total_time)
            point.time_from_start.nanosec = int((total_time - int(total_time)) * 1e9)
            traj_msg.points.append(point)

        self.trajectory_pub.publish(traj_msg)
        self.node.get_logger().info(
            f"发送完整轨迹：{num_points}个点，总时长{total_duration}s")

        # 4. 更新状态并等待执行完毕
        self.current_gripper = gripper_position
        with self.joint_state_lock:
            self.current_joint_positions = joint_waypoints[-1]
        self.current_position = end_pos

        time.sleep(total_duration + 0.5)
        return True

    def move_in_arc_simple(self, start_pos: List[float], end_pos: List[float], radius: float = 0.2,
                           clockwise: bool = True, gripper_position: Optional[float] = None,
                           total_duration: float = 2.0, num_points: int = 8) -> bool:
        """
        水平面内圆弧移动(逐点发送):夹持物体长距离转移时,
        相比关节空间解,水平圆弧的IK解通常更合理平稳。
        """
        if gripper_position is None:
            gripper_position = self.current_gripper

        plane_z = start_pos[2]

        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        chord_length = np.sqrt(dx * dx + dy * dy)

        if chord_length < 0.001:
            self.node.get_logger().info("距离太近，直接移动")
            return self.move_to_position(end_pos, gripper_position, total_duration)

        if radius is None:
            radius = chord_length * 0.7
        if radius < chord_length / 2:
            radius = chord_length / 2

        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2
        chord_dir_x = dx / chord_length
        chord_dir_y = dy / chord_length

        h_squared = radius ** 2 - (chord_length / 2) ** 2
        if h_squared < 0:
            if abs(h_squared) < 0.001:
                h = 0.0
            else:
                self.node.get_logger().error(f"无法计算圆弧，h_squared={h_squared}")
                return self.move_linearly(start_pos, end_pos, 3, gripper_position,
                                          total_duration / 3)
        else:
            h = np.sqrt(h_squared)

        if clockwise:
            center_x = mid_x - chord_dir_y * h
            center_y = mid_y + chord_dir_x * h
        else:
            center_x = mid_x + chord_dir_y * h
            center_y = mid_y - chord_dir_x * h

        start_angle = np.arctan2(start_pos[1] - center_y, start_pos[0] - center_x)
        end_angle = np.arctan2(end_pos[1] - center_y, end_pos[0] - center_x)

        if clockwise and end_angle > start_angle:
            end_angle -= 2 * np.pi
        if not clockwise and end_angle < start_angle:
            end_angle += 2 * np.pi

        waypoints = []
        for angle in np.linspace(start_angle, end_angle, num_points + 1):
            waypoints.append([
                center_x + radius * np.cos(angle),
                center_y + radius * np.sin(angle),
                plane_z
            ])

        self.node.get_logger().info(f"圆弧移动: 半径={radius:.3f}m, 点={len(waypoints)}个")

        point_duration = total_duration / len(waypoints)
        for i, pos in enumerate(waypoints):
            if not self.move_to_position_direct(pos, gripper_position, point_duration):
                self.node.get_logger().warn(f"点 {i} 移动失败，尝试继续")
                continue
            time.sleep(point_duration * 0.1)

        return True

    # ------------------------------------------------------------------
    # 夹爪与抓放流程
    # ------------------------------------------------------------------
    def control_gripper(self, open: bool = True, width: Optional[float] = None,
                        duration: float = 1.0) -> bool:
        """控制夹爪(保持当前关节角,仅改夹爪;width 语义见文件头)"""
        if width is None:
            width = self.gripper_open if open else self.gripper_close

        gripper_pos = 0.8 - width
        return self.send_joint_command(
            self.get_current_joint_positions(),
            gripper_position=gripper_pos,
            duration=duration
        )

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

    def _arc_move_with_fallback(self, start_pos, end_pos, safe_pos, clockwise,
                                total_duration) -> bool:
        """圆弧移动,失败时回退直线(抓放流程中两处相同逻辑的公共实现)"""
        try:
            if self.move_in_arc_simple(start_pos=start_pos, end_pos=end_pos,
                                       radius=self.arc_radius, clockwise=clockwise,
                                       total_duration=total_duration):
                return True
            self.node.get_logger().warn("圆弧移动失败，使用直线移动")
        except Exception as e:
            self.node.get_logger().error(f"圆弧移动异常: {e}，回退直线移动")
        return self.move_linearly(safe_pos, end_pos, 5, duration_per_step=0.5)

    def execute_pick_place(self, object_position: List[float],
                           place_position: Optional[List[float]] = None,
                           object_type: str = "unknown", object_size: float = 0.03) -> bool:
        """
        完整抓取-放置流程:
        归位 → 张爪 → 接近 → 下降 → 合爪 → 提升 → 圆弧转移 → 下降放置
        → 松爪 → 提升 → 圆弧返回归位
        """
        if place_position is None:
            place_position = self.default_place_position

        self.node.get_logger().info("开始抓取放置操作")

        try:
            # 0. 归位
            if not self.go_home():
                return False

            # 1. 打开夹爪
            self.node.get_logger().info("1. 打开夹爪")
            self.control_gripper(open=True, width=0.8)

            # 2. 移动到物体上方
            above_pos = [object_position[0], object_position[1],
                         object_position[2] + self.approach_height]
            self.node.get_logger().info(f"2. 移动到物体上方: {above_pos}")
            if not self.move_to_position_direct(above_pos, duration=1.5):
                return False
            time.sleep(1)

            # 3. 直线下降到抓取位
            grasp_pos = [object_position[0], object_position[1],
                         object_position[2] + self.pick_height]
            self.node.get_logger().info(f"3. 下降到抓取位置: {grasp_pos}")
            if not self.move_linearly(above_pos, grasp_pos, 5, duration_per_step=0.3):
                return False

            # 4. 闭合夹爪
            grasp_width = self._get_grasp_width(object_type, object_size)
            self.node.get_logger().info(f"4. 关闭夹爪 (宽度={grasp_width:.3f}m)")
            self.control_gripper(open=False, width=grasp_width, duration=1.0)

            # 5. 提升
            self.node.get_logger().info("5. 提升物体")
            if not self.move_linearly(grasp_pos, above_pos, 3, duration_per_step=0.2):
                self.control_gripper(open=True)
                return False

            self.go_home()
            time.sleep(1)

            # 6. 水平圆弧转移到放置点上方(安全高度)
            safe_pos = [object_position[0], object_position[1], self.safe_height]
            place_safe = [place_position[0], place_position[1], self.safe_height]
            self.node.get_logger().info("6. 圆弧移动到放置位置")
            if not self._arc_move_with_fallback(self.home_position, place_safe,
                                                safe_pos, clockwise=True,
                                                total_duration=5.0):
                self.control_gripper(open=True)
                return False
            time.sleep(1)

            # 7. 下降到放置点上方
            place_above = [place_position[0], place_position[1],
                           place_position[2] + self.approach_height]
            self.node.get_logger().info(f"7. 下降到放置位置: {place_above[2]:.3f}m")
            if not self.move_to_position_direct(place_above, duration=1.0):
                self.control_gripper(open=True)
                return False
            time.sleep(1)

            # 8. 下降到放置点
            place_done = [place_position[0], place_position[1],
                          place_position[2] + self.pick_height]
            if not self.move_to_position_direct(place_done, duration=0.8):
                self.control_gripper(open=True)
                return False
            time.sleep(1)

            # 9. 松爪
            self.node.get_logger().info("9. 放置物体")
            self.control_gripper(open=True, width=0.8, duration=1.0)
            time.sleep(1)

            # 10. 提升离开
            self.node.get_logger().info("10. 提升离开")
            if not self.move_to_position_direct(place_safe, duration=0.8):
                self.node.get_logger().warn("提升离开失败")
            time.sleep(1)

            # 11. 圆弧返回归位
            self.node.get_logger().info("11. 圆弧返回初始位置")
            if not self._arc_move_with_fallback(place_safe, self.home_position,
                                                place_safe, clockwise=False,
                                                total_duration=3.0):
                self.control_gripper(open=True)
                return False

            time.sleep(1)
            self.node.get_logger().info("✅ 抓取放置完成")
            return True

        except Exception as e:
            self.node.get_logger().error(f"抓取放置失败: {str(e)}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())
            try:
                self.control_gripper(open=True)
                self.go_home()
            except Exception:
                pass
            return False

    def _get_grasp_width(self, object_type: str, object_size: float) -> float:
        """按物体类型估算合爪宽度"""
        if object_type in ["bottle", "cup", "cylinder"]:
            width = object_size * 0.9
        elif object_type in ["box", "cube"]:
            width = object_size
        elif object_type in ["sphere", "ball"]:
            width = object_size * 1.1
        else:
            width = object_size * 0.9
        return max(0.01, min(width, self.gripper_open))

    def get_status(self):
        return {
            'position': self.current_joint_positions,
            'gripper': self.current_gripper,
            'orientation': 'down',
            'state': 'home' if self.current_position == self.home_position else 'working'
        }
