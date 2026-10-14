#!/usr/bin/env python3
"""无头测试:MoveItPy 能否在本环境初始化并完成一次规划(不需要 Isaac Sim)"""
import sys
import yaml

# 1. 用 MoveItConfigsBuilder 组装 ur5 配置并写成 params 文件
from moveit_configs_utils import MoveItConfigsBuilder

mc = (MoveItConfigsBuilder(robot_name="ur5", package_name="ur5_moveit_config")
      .robot_description(file_path="config/ur5.urdf.xacro")
      .robot_description_semantic(file_path="config/ur5.srdf")
      .robot_description_kinematics(file_path="config/kinematics.yaml")
      .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
      .to_moveit_configs())
cfg = mc.to_dict()
print("config keys:", sorted(cfg.keys()))

# 修复:MoveItConfigsUtils 生成的 ompl 段是双层嵌套(ompl.ompl.planning_plugin),
# 而本版本 moveit_cpp 的 PlanningPipeline 读取 ompl.planning_plugin → 展平一层
if isinstance(cfg.get("ompl"), dict) and isinstance(cfg["ompl"].get("ompl"), dict):
    inner = cfg["ompl"]["ompl"]
    outer_extras = {k: v for k, v in cfg["ompl"].items() if k != "ompl"}
    inner.update({k: v for k, v in outer_extras.items() if k not in inner})
    cfg["ompl"] = inner
    print("ompl 段已展平, planning_plugin =", cfg["ompl"].get("planning_plugin"))

# 修复2:moveit_cpp 读取的是 planning_pipelines.pipeline_names(命名空间形式),
# 而 builder 生成的是普通列表 → 转成 moveit_cpp.yaml 的格式
if isinstance(cfg.get("planning_pipelines"), list):
    cfg["planning_pipelines"] = {
        "pipeline_names": cfg["planning_pipelines"],
        "default_planning_pipeline": cfg.get("default_planning_pipeline", "ompl"),
    }
    print("planning_pipelines 已转为命名空间形式:", cfg["planning_pipelines"])
# 2. 把完整配置字典通过 config_dict 传给 MoveItPy(C++ 侧会生成 params 文件)
from moveit.planning import MoveItPy

robot = MoveItPy(node_name="moveit_py_test", config_dict=cfg)
print("✅ MoveItPy 初始化成功")

arm = robot.get_planning_component("ur5_arm")
print("✅ planning component 获取成功, named states:", arm.named_target_states)

# 3. 用命名状态设起点(SRDF 中的 home_pose),做一次关节目标规划
arm.set_start_state(configuration_name="home_pose")
arm.set_goal_state(configuration_name="home_pose")


from moveit.planning import PlanRequestParameters

PR = PlanRequestParameters(robot)  # 本版本从 MoveItPy 实例构造,再逐字段赋值
PR.planning_pipeline = "ompl"
PR.planner_id = "RRTConnectkConfigDefault"
PR.planning_attempts = 5
PR.planning_time = 5.0
PR.max_velocity_scaling_factor = 0.5
PR.max_acceleration_scaling_factor = 0.5


def try_plan():
    try:
        return arm.plan(parameters=PR)  # 本版本 plan() 需要显式 pipeline 名
    except TypeError:
        return arm.plan()  # 新版本 plan() 默认参数即可


sol = try_plan()
if sol:
    rt = sol.trajectory
    traj = rt.get_robot_trajectory_msg().joint_trajectory  # RobotTrajectory → JointTrajectory msg
    print(f"✅ 规划成功: {len(traj.joint_names)} 关节, {len(traj.points)} 轨迹点")
    print("joint_names:", traj.joint_names)
    t0, tn = traj.points[0], traj.points[-1]
    print("首点:", [round(v, 3) for v in t0.positions],
          "t =", t0.time_from_start.sec + t0.time_from_start.nanosec * 1e-9)
    print("末点:", [round(v, 3) for v in tn.positions],
          "t =", tn.time_from_start.sec + tn.time_from_start.nanosec * 1e-9)
else:
    print("❌ 规划失败, error_code:", getattr(sol, "error_code", "n/a"))
    sys.exit(2)

# 4. 位姿目标规划(验证 IK+OMPL 全链路)
from geometry_msgs.msg import PoseStamped
goal = PoseStamped()
goal.header.frame_id = "world"
goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = 0.3, 0.3, 0.5
goal.pose.orientation.w = 1.0
arm.set_goal_state(pose_stamped_msg=goal, pose_link="wrist_3_link")
sol2 = try_plan()
if sol2:
    rt2 = sol2.trajectory
    traj2 = rt2.get_robot_trajectory_msg().joint_trajectory
    tn = traj2.points[-1]
    print(f"✅ 位姿目标规划成功: {len(traj2.points)} 点, 末点:",
          [round(v, 3) for v in tn.positions])
else:
    print("❌ 位姿目标规划失败(目标可能不可达)")
    sys.exit(3)

print("ALL_OK")
