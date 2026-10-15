#!/bin/bash
# 全流程抓取测试(在 moveit_aluupy 容器内运行,前置:arm-wb-sim 仿真已稳定)
# 组成:robot_state_publisher(TF) + move_group + 转换器 + 观察视频录制 + main.py --once
set -x
source /opt/ros/humble/setup.bash
source /ros2_ws/moveit_ws/install/setup.bash 2>/dev/null
export PYTHONPATH=/ros2_ws/moveit_ws/install/moveit_py/local/lib/python3.10/dist-packages:$PYTHONPATH
export ROS_DOMAIN_ID=77 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export QT_QPA_PLATFORM=offscreen
cd /ros2_ws/isaacsim_arm_vision

# 0. URDF(缺则由 xacro 生成)
URDF=/ros2_ws/moveit_ws/ur5.urdf
[ -f "$URDF" ] || ros2 run xacro xacro /ros2_ws/moveit_ws/src/ur5_moveit_config/config/ur5.urdf.xacro > "$URDF"

# 1. robot_state_publisher:订阅 /joint_states → 发布 TF
ros2 run robot_state_publisher robot_state_publisher \
    --ros-args -p robot_description:="$(cat $URDF)" > /tmp/rsp.log 2>&1 &
RSP=$!
sleep 2

# 2. move_group + 轨迹转换器
ros2 launch ur5_moveit_config start_moveit_isaac.launch.py > /tmp/mg.log 2>&1 &
MG=$!
sleep 15

# 3. 观察相机视频录制
REC_SECONDS=${REC_SECONDS:-240}
timeout -s INT $REC_SECONDS python3 /tmp/recorder.py /observer_rgb \
    /ros2_ws/isaacsim_arm_vision/detection_results/observer.mp4 15 > /tmp/rec.log 2>&1 &
REC=$!

# 4. 单次检测+抓取(MoveItPy + 两阶段 + 边缘朝向)
python3 main.py --once 2>&1 | tee /tmp/main_once.log
MAIN_RC=${PIPESTATUS[0]}

# 5. 收尾:多录几秒后停录制
sleep 8
kill -INT $REC 2>/dev/null
sleep 3
kill $MG $RSP 2>/dev/null
sleep 2
kill -9 $MG $RSP 2>/dev/null

echo "MAIN_RC=$MAIN_RC"
echo "TEST_DONE"
