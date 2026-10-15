#!/usr/bin/env python3
"""
MoveItPy 版机械臂控制器(分段式执行)

分工:
  - 自由空间点对点运动(归位、转移、接近)→ MoveItPy/OMPL 整条轨迹规划,
    带时间最优参数化,取代手写圆弧与逐点发送;
  - 垂直下降/提升 → 沿用父类 move_linearly(直线插值 + 逐点IK + 整条轨迹一次发布),
    保证抓取时的垂直接近方向;
  - 夹爪 → 沿用父类单点命令。

执行链路:轨迹发布到 /moveit_trajectory,由 ur5_moveit_config/launch/trajectory_converter.py
(升级版)按时间戳插值回放到 /joint_command → Isaac Sim。

本文件同时封装了本版本 moveit_py(2.5.x)初始化所需的三个配置修复,详见
build_ur5_moveit_config() 注释——这些是当初 MoveItPy 方案跑不起来的真正原因。
"""
import time
from typing import List, Optional

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped

from modules.arm_manager import ArmControllerWithGripper


def build_ur5_moveit_config() -> dict:
    """
    组装 ur5_moveit_config 的完整参数字典,并修复与本版本 moveit_cpp 的格式差异:

    1. MoveItConfigsUtils 生成的 ompl 段是双层嵌套(ompl.ompl.planning_plugin),
       而 moveit_cpp 的 PlanningPipeline 读取 ompl.planning_plugin → 展平一层;
    2. moveit_cpp 读取 planning_pipelines.pipeline_names(命名空间形式),
       builder 生成的是普通列表 → 转换;
    3. MoveItPy 的 config_dict 会被写成 params 文件喂给它自己的节点,
       传空字典等于没有任何机器人描述 → 必须传完整配置。
    """
    from moveit_configs_utils import MoveItConfigsBuilder

    mc = (MoveItConfigsBuilder(robot_name="ur5", package_name="ur5_moveit_config")
          .robot_description(file_path="config/ur5.urdf.xacro")
          .robot_description_semantic(file_path="config/ur5.srdf")
          .robot_description_kinematics(file_path="config/kinematics.yaml")
          .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
          .to_moveit_configs())
    cfg = mc.to_dict()

    if isinstance(cfg.get("ompl"), dict) and isinstance(cfg["ompl"].get("ompl"), dict):
        inner = cfg["ompl"]["ompl"]
        for k, v in cfg["ompl"].items():
            if k != "ompl" and k not in inner:
                inner[k] = v
        cfg["ompl"] = inner

    if isinstance(cfg.get("planning_pipelines"), list):
        cfg["planning_pipelines"] = {
            "pipeline_names": cfg["planning_pipelines"],
            "default_planning_pipeline": cfg.get("default_planning_pipeline", "ompl"),
        }

    return cfg


class MoveItArmController(ArmControllerWithGripper):
    """MoveItPy 规划版控制器,接口与 ArmControllerWithGripper 完全兼容"""

    def __init__(self, node, params: Optional[dict] = None, moveit_config: Optional[dict] = None):
        super().__init__(node, params)

        # MoveItPy 相关导入放在类内,保证 ik 模式无需安装 moveit_py
        from moveit.planning import MoveItPy, PlanRequestParameters

        cfg = moveit_config or build_ur5_moveit_config()
        self.moveit = MoveItPy(node_name="moveit_py_arm", config_dict=cfg)
        self.planning_component = self.moveit.get_planning_component("ur5_arm")

        self.plan_params = PlanRequestParameters(self.moveit)
        self.plan_params.planning_pipeline = "ompl"
        self.plan_params.planner_id = "RRTConnectkConfigDefault"
        self.plan_params.planning_attempts = 5
        self.plan_params.planning_time = 5.0
        self.plan_params.max_velocity_scaling_factor = 0.5
        self.plan_params.max_acceleration_scaling_factor = 0.5

        self.node.get_logger().info("✅ MoveItPy 控制器初始化成功(OMPL 规划已启用)")

    # ------------------------------------------------------------------
    # 规划与轨迹发送
    # ------------------------------------------------------------------
    def _plan_to_pose(self, position: List[float], orientation: List[float]) -> Optional[JointTrajectory]:
        """规划到目标位姿,返回 6 关节 JointTrajectory(含时间参数化),失败返回 None"""
        goal = PoseStamped()
        goal.header.frame_id = "world"
        goal.header.stamp = self.node.get_clock().now().to_msg()
        goal.pose.position.x = position[0]
        goal.pose.position.y = position[1]
        goal.pose.position.z = position[2]
        goal.pose.orientation.x = orientation[0]
        goal.pose.orientation.y = orientation[1]
        goal.pose.orientation.z = orientation[2]
        goal.pose.orientation.w = orientation[3]

        try:
            self.planning_component.set_start_state_to_current_state()
            self.planning_component.set_goal_state(
                pose_stamped_msg=goal, pose_link="wrist_3_link")
            sol = self.planning_component.plan(parameters=self.plan_params)
            if not sol:
                self.node.get_logger().warn(f"OMPL 规划失败: {position}")
                return None
            return sol.trajectory.get_robot_trajectory_msg().joint_trajectory
        except Exception as e:
            self.node.get_logger().warn(f"MoveIt 规划异常: {e}")
            return None

    def _send_trajectory(self, traj: JointTrajectory, gripper_position: Optional[float] = None,
                         extra_wait: float = 0.5) -> bool:
        """发布整条轨迹到 /moveit_trajectory(附加 finger_joint),按轨迹时长等待执行"""
        if gripper_position is None:
            gripper_position = self.current_gripper

        msg = JointTrajectory()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.joint_names = list(traj.joint_names) + ['finger_joint']

        total_time = 0.0
        for pt in traj.points:
            new_pt = JointTrajectoryPoint()
            new_pt.positions = list(pt.positions) + [gripper_position]
            new_pt.time_from_start = pt.time_from_start
            msg.points.append(new_pt)
            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            total_time = max(total_time, t)

        self.trajectory_pub.publish(msg)
        self.node.get_logger().info(
            f"发送 MoveIt 轨迹: {len(msg.points)} 点, 时长 {total_time:.2f}s")

        # 更新本地估计并等待执行完毕
        self.current_gripper = gripper_position
        with self.joint_state_lock:
            self.current_joint_positions = list(traj.points[-1].positions)
        time.sleep(total_time + extra_wait)
        return True

    # ------------------------------------------------------------------
    # 覆写运动原语
    # ------------------------------------------------------------------
    def move_to_position_direct(self, position: List[float], gripper_position: Optional[float] = None,
                                duration: float = 1.0, orientation_type: str = "down",
                                orientation: Optional[List[float]] = None) -> bool:
        """自由空间运动:MoveItPy 规划整条轨迹;失败时回退到父类的 IK 单点模式"""
        orient = self._resolve_orientation(orientation_type, orientation)
        traj = self._plan_to_pose(position, orient)
        if traj is not None and len(traj.points) > 0:
            self.current_position = list(position)
            return self._send_trajectory(traj, gripper_position)

        self.node.get_logger().warn("MoveIt 规划失败,回退到 IK 单点模式")
        return super().move_to_position_direct(position, gripper_position, duration,
                                                orientation_type, orientation)

    def move_in_arc_simple(self, start_pos: List[float], end_pos: List[float], radius: float = 0.2,
                           clockwise: bool = True, gripper_position: Optional[float] = None,
                           total_duration: float = 2.0, num_points: int = 8) -> bool:
        """
        有了真正的规划器,不再需要手写圆弧来规避不合理的自由空间解——
        直接交给 OMPL 点对点规划。保留方法签名以兼容 execute_pick_place 的调用。
        """
        return self.move_to_position_direct(end_pos, gripper_position)

    def shutdown(self):
        """清理 MoveItPy(先于 rclpy 关闭)"""
        try:
            self.moveit.shutdown()
        except Exception:
            pass
