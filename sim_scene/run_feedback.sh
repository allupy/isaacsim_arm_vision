#!/bin/bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /ros2_ws/moveit_ws/install/setup.bash
export PYTHONPATH=/ros2_ws/moveit_ws/install/moveit_py/local/lib/python3.10/dist-packages:${PYTHONPATH:-}
cd /experiment/src
exec python3 -u feedback_runner.py "$@"
