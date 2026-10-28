#!/bin/bash
set -euo pipefail
ROOT=/home/aluupy/arm_feedback_experiments/20261027
docker run -d --name arm-feedback-control-20261027 --entrypoint bash \
 --gpus '"device=3"' --network=host --ipc=private \
 -e ROS_DOMAIN_ID=78 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
 -e FASTRTPS_DEFAULT_PROFILES_FILE=/experiment/src/sim_scene/fastdds_udp.xml \
 -e QT_QPA_PLATFORM=offscreen \
 -v /home/aluupy/omnilrs_ros2_demo/humble_ws:/ros2_ws:ro \
 -v "$ROOT:/experiment" moveit_aluupy:latest \
 /experiment/src/sim_scene/control_services.sh
