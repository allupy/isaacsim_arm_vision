#!/usr/bin/env python3
"""端到端测试:MoveItPy 控制器 → /moveit_trajectory → 轨迹转换器 → /joint_command
前置:move_group 与升级版 trajectory_converter 已在运行(同一 ROS_DOMAIN_ID)
不需要 Isaac Sim——用假 /joint_states 代替。
"""
import sys
import time
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

sys.path.insert(0, '/ros2_ws/isaacsim_arm_vision')
from modules.moveit_controller import MoveItArmController

NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
         'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 'finger_joint']
HOME = [-0.729, -1.6315, 1.5447, 0.0, 0.0, 0.0, 0.0]

rclpy.init()
node = Node('test_e2e')

# 1. 模拟 Isaac Sim 发布 /joint_states(20Hz,保持在 HOME)
js_pub = node.create_publisher(JointState, '/joint_states', 10)


def pub_js():
    js = JointState()
    js.header.stamp = node.get_clock().now().to_msg()
    js.name = NAMES
    js.position = list(HOME)
    js_pub.publish(js)


node.create_timer(0.05, pub_js)

# 2. 记录转换器输出 /joint_command
cmds = []
t_start = time.monotonic()


def cb_cmd(msg):
    cmds.append((time.monotonic() - t_start, list(msg.position)))


node.create_subscription(JointState, '/joint_command', cb_cmd, 10)

# 3. 构造控制器
ctrl = MoveItArmController(node)
print("✅ 控制器构造成功")

# 后台 spin(回调持续运行)
spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
spin_thread.start()
time.sleep(1.0)

# 4. 执行一次自由空间运动(OMPL 规划 → 轨迹发布 → 转换器回放)
t_start = time.monotonic()
cmds.clear()
ok = ctrl.move_to_position([0.3, 0.3, 0.5], duration=3.0)
print("move_to_position 返回:", ok)

time.sleep(1.0)
print(f"收到 /joint_command 消息数: {len(cmds)}")
if len(cmds) >= 5:
    first, last = cmds[0], cmds[-1]
    print("首帧关节:", [round(v, 3) for v in first[1][:6]])
    print("末帧关节:", [round(v, 3) for v in last[1][:6]])
    print(f"回放时间跨度: {last[0] - first[0]:.2f}s")
    print("E2E_OK" if last[0] - first[0] > 0.5 else "E2E_FAIL: 回放过快,疑似只发了单点")
else:
    print("E2E_FAIL: 回放消息过少")

# 5. 夹爪单点命令兼容性
ctrl.control_gripper(open=False, width=0.03, duration=0.5)
time.sleep(0.5)
print(f"夹爪命令后消息总数: {len(cmds)}")

# moveit_py 析构在部分环境会 abort,直接以结果退出
import os
playback_ok = len([c for c in cmds]) >= 5
os._exit(0 if playback_ok else 2)
