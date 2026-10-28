#!/bin/bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /ros2_ws/moveit_ws/install/setup.bash
export PYTHONPATH=/ros2_ws/moveit_ws/install/moveit_py/local/lib/python3.10/dist-packages:${PYTHONPATH:-}
cd /experiment/src
exec ros2 launch /experiment/src/sim_scene/feedback_moveit.launch.py > /experiment/move_group.log 2>&1
