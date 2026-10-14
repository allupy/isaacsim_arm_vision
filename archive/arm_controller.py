#!/usr/bin/env python3
# moveit2_arm_controller.py
"""
机械臂控制器 - 使用MoveIt 2 Python API,似乎机械臂的配置文件有问题，没有办法进行初始化
"""
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose, PoseStamped
import sys
import math
import time
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState


class MoveItArmController(Node):
    def __init__(self, node: Node = None):
        if node is None:
            super().__init__('moveit2_arm_controller')
            self.own_node = True
        else:
            self.node = node
            self.own_node = False
        
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
        
        # 轨迹发布者（用于夹爪控制）
        self.trajectory_pub = self.node.create_publisher(
            JointTrajectory,
            '/moveit_trajectory',
            10
        )
        
        # 初始化MoveIt 2
        self.moveit = None
        self.planning_component = None
        self.init_moveit2()
        
        # 夹爪参数
        self.current_gripper = 0.08
        self.gripper_open = 0.08
        self.gripper_close = 0.0
        self.grasp_force = 0.4
        
        # 抓取参数
        self.pick_height = 0.02
        self.approach_height = 0.15
        
        # 当前位置
        self.current_position = [0.5, 0.36, 0.2]
        
        self.node.get_logger().info(f"✅ MoveIt 2控制器初始化完成")
    
    def init_moveit2(self):
        """正确的MoveItPy初始化方法"""
        try:
            from moveit.planning import MoveItPy
            
            self.moveit = MoveItPy(
                node_name="moveit_py",  # 通常moveit_py是标准的
                config_dict={},  # 空配置，因为moveit已经在运行
                launch_params_filepaths=[],
                provide_planning_service=False  # 不启动新服务，连接到现有的
            )
            
                        
            # 获取规划组件 - 这里的group_name需要和你MoveIt配置中的规划组一致
            # 在官方教程中是 panda_arm = panda.get_planning_component("arm")
            # 你需要改为你的规划组名称，比如"jetarm"或"arm"
            self.planning_component = self.moveit.get_planning_component("ur5_arm")  # 修改为你的规划组名称
            
            # 获取机器人模型
            self.robot_model = self.moveit.get_robot_model()
            
            self.node.get_logger().info("✅ MoveIt 2初始化成功")
            
            
        except Exception as e:
            self.node.get_logger().error(f"MoveIt 2初始化失败: {e}")
    
    def plan_and_execute(self, single_plan_parameters=None, multi_plan_parameters=None):
        """规划并执行运动"""
        if self.planning_component is None:
            self.node.get_logger().error("❌ 规划组件未初始化")
            return False
        
        try:
            self.node.get_logger().info("正在规划轨迹...")
            
            # 规划
            if multi_plan_parameters is not None:
                plan_result = self.planning_component.plan(
                    multi_plan_parameters=multi_plan_parameters
                )
            elif single_plan_parameters is not None:
                plan_result = self.planning_component.plan(
                    single_plan_parameters=single_plan_parameters
                )
            else:
                plan_result = self.planning_component.plan()
            
            # 执行
            if plan_result:
                self.node.get_logger().info("执行计划")
                robot_trajectory = plan_result.trajectory
                self.moveit.execute(robot_trajectory, controllers=[])
                return True
            else:
                self.node.get_logger().error("❌ 规划失败")
                return False
                
        except Exception as e:
            self.node.get_logger().error(f"规划执行失败: {e}")
            return False
    
    def move_to_pose(self, position, orientation=None, wait=True):
        """移动到目标位姿"""
        if self.planning_component is None:
            self.node.get_logger().error("❌ 规划组件未初始化")
            return False
        
        try:
            # 设置起始状态为当前状态
            self.planning_component.set_start_state_to_current_state()
            
            # 创建姿态目标
            pose_goal = PoseStamped()
            pose_goal.header.frame_id = "world"
            pose_goal.header.stamp = self.node.get_clock().now().to_msg()
            
            if orientation is None:
                orientation = self.get_down_orientation()
            
            pose_goal.pose.position.x = position[0]
            pose_goal.pose.position.y = position[1]
            pose_goal.pose.position.z = position[2]
            
            pose_goal.pose.orientation.x = orientation[0]
            pose_goal.pose.orientation.y = orientation[1]
            pose_goal.pose.orientation.z = orientation[2]
            pose_goal.pose.orientation.w = orientation[3]
            
            # 设置目标
            self.planning_component.set_goal_state(
                pose_stamped_msg=pose_goal, 
                pose_link="wrist_3_link"  # 末端执行器链路
            )
            
            # 规划并执行
            self.node.get_logger().info(f"规划到位置: {position}")
            success = self.plan_and_execute()
            
            if success:
                self.current_position = position
                self.node.get_logger().info("✅ 移动完成")
            else:
                self.node.get_logger().warn("⚠️ 移动失败")
            
            return success
            
        except Exception as e:
            self.node.get_logger().error(f"移动失败: {e}")
            return False
    
    def get_down_orientation(self):
        """获取竖直向下朝向 (0, -90°, 0)"""
        # ROS 2中标准的竖直向下四元数
        return [0.0, -0.7071, 0.0, 0.7071]
    
    def control_gripper(self, open=True, width=None, duration=1.0):
        """控制夹爪"""
        if width is None:
            width = self.gripper_open if open else self.gripper_close
        
        # 获取当前关节位置
        current_joints = []
        if self.moveit:
            try:
                # 获取当前机器人状态
                robot_state = RobotState(self.moveit.get_robot_model())
                # 这里需要根据实际情况获取当前关节位置
                # 简化处理，使用默认值
                current_joints = [0.0] * 6
            except:
                current_joints = [0.0] * 6
        
        # 发送关节命令
        joint_positions = list(current_joints) + [width]
        
        msg = JointTrajectory()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = 0
        
        msg.points = [point]
        self.trajectory_pub.publish(msg)
        
        # 更新状态
        self.current_gripper = width
        
        state = "打开" if open else "关闭"
        self.node.get_logger().info(f"夹爪{state}: 位置={width:.3f}")
        
        time.sleep(duration)
        return True
    
    def go_home(self, wait=True):
        """回到初始位置"""
        home_position = [0.5, 0.36, 0.2]
        orientation = self.get_down_orientation()
        
        self.node.get_logger().info("回到初始位置...")
        
        success = self.move_to_pose(home_position, orientation, wait)
        
        if success:
            self.node.get_logger().info("✅ 已回到初始位置")
        else:
            self.node.get_logger().error("❌ 回到初始位置失败")
        
        return success
    
    def execute_pick_place(self, object_position, place_position=None, 
                          object_type="unknown", object_size=0.03):
        """执行抓取放置操作"""
        if place_position is None:
            place_position = [0.3, 0.2, 0.1]
        
        self.node.get_logger().info("🤖 开始抓取操作")
        self.node.get_logger().info(f"物体位置: {object_position}")
        
        try:
            # 1. 准备工作
            self.node.get_logger().info("1. 准备工作")
            self.go_home()
            time.sleep(1)
            
            # 2. 打开夹爪
            self.node.get_logger().info("2. 打开夹爪")
            self.control_gripper(open=True, width=object_size + 0.02)
            time.sleep(1)
            
            # 3. 移动到物体上方
            above_pos = [object_position[0], object_position[1], 
                        object_position[2] + self.approach_height]
            self.node.get_logger().info(f"3. 移动到物体上方: {above_pos}")
            
            if not self.move_to_pose(above_pos):
                return False
            time.sleep(1)
            
            # 4. 下降到抓取位置
            grasp_pos = [object_position[0], object_position[1], 
                        object_position[2] + self.pick_height]
            self.node.get_logger().info(f"4. 下降到抓取位置: {grasp_pos}")
            
            if not self.move_to_pose(grasp_pos):
                return False
            time.sleep(1)
            
            # 5. 抓取物体
            self.node.get_logger().info("5. 关闭夹爪")
            grasp_width = self._get_grasp_width(object_type, object_size)
            self.control_gripper(open=False, width=grasp_width)
            time.sleep(2)
            
            # 6. 提升物体
            self.node.get_logger().info("6. 提升物体")
            if not self.move_to_pose(above_pos):
                return False
            time.sleep(1)
            
            # 7. 移动到放置位置
            place_above = [place_position[0], place_position[1], 
                          place_position[2] + self.approach_height]
            self.node.get_logger().info(f"7. 移动到放置位置上方: {place_above}")
            if not self.move_to_pose(place_above):
                return False
            time.sleep(1)
            
            # 8. 放置物体
            self.node.get_logger().info(f"8. 下降到放置位置: {place_position}")
            if not self.move_to_pose(place_position):
                return False
            time.sleep(1)
            
            # 9. 释放物体
            self.node.get_logger().info("9. 释放物体")
            self.control_gripper(open=True)
            time.sleep(1)
            
            # 10. 回到安全位置
            self.node.get_logger().info("10. 回到安全位置")
            self.go_home()
            
            self.node.get_logger().info("✅ 抓取操作完成")
            return True
            
        except Exception as e:
            self.node.get_logger().error(f"抓取失败: {e}")
            # 出错时打开夹爪并回到安全位置
            self.control_gripper(open=True)
            self.go_home()
            return False
    
    def _get_grasp_width(self, object_type, object_size):
        """获取合适的抓取宽度"""
        if object_type in ["bottle", "cup", "cylinder"]:
            return object_size - 0.01
        elif object_type in ["box", "cube"]:
            return object_size
        else:
            return object_size * 0.9
    
    def get_current_pose(self):
        """获取当前末端位姿"""
        if self.planning_component is None:
            return {
                'position': self.current_position,
                'orientation': self.get_down_orientation()
            }
        
        # 这里需要根据MoveIt 2的API获取当前位姿
        # 简化处理，返回当前位置
        return {
            'position': self.current_position,
            'orientation': self.get_down_orientation()
        }
    
    def get_current_joints(self):
        """获取当前关节位置"""
        if self.planning_component is None:
            return [0.0] * 6
        
        # 这里需要根据MoveIt 2的API获取当前关节位置
        # 简化处理，返回默认值
        return [0.0] * 6
    
    def get_status(self):
        """获取状态"""
        return {
            'position': self.current_position,
            'gripper': self.current_gripper,
            'moveit_initialized': self.moveit is not None,
            'planning_component_available': self.planning_component is not None
        }
    
    def cleanup(self):
        """清理资源"""
        if rclpy.ok():
            rclpy.shutdown()