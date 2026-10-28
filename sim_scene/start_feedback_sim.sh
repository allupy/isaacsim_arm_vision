#!/bin/bash
set -euo pipefail
ROOT=/home/aluupy/arm_feedback_experiments/20261027
SCENE=/mnt/intel_ssd/omnilrs/OmniLRS/robots
mkdir -p "$ROOT/sim"
if test ! -e "$ROOT/sim/runtime_config.json"; then
  cp "$ROOT/src/sim_scene/runtime_optimized.json" "$ROOT/sim/runtime_config.json"
fi
docker run -d --name "${SIM_NAME:-arm-feedback-sim-20261027}" --entrypoint bash \
  --gpus '"device=2"' --network=host --ipc=private --shm-size=8g \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e ROS_DOMAIN_ID=78 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/tmp/udp_only.xml \
  -e SIM_DURATION=${SIM_DURATION:-3600} -e CAM_EVERY=1 \
  -e REPAIR_PARKED_POSE=${REPAIR_PARKED_POSE:-1} -e REPAIR_CAMERA_MOUNT=${REPAIR_CAMERA_MOUNT:-1} \
  -e PHYSICS_STEPS_PER_RENDER=${PHYSICS_STEPS_PER_RENDER:-120} \
  -v "$SCENE:/ws:ro" -v "$ROOT:/experiment" \
  -v /home/aluupy/docker/isaac-sim/cache_wb/ov:/root/.cache/ov:rw \
  -v /home/aluupy/docker/isaac-sim/cache_wb/glcache:/root/.cache/nvidia/GLCache:rw \
  -v /home/aluupy/docker/isaac-sim/cache_wb/computecache:/root/.nv/ComputeCache:rw \
  -v "$ROOT/src/sim_scene/fastdds_udp.xml:/tmp/udp_only.xml:ro" \
  isaac-sim-omnilrs-0513:latest \
  -c '/isaac-sim/python.sh /experiment/src/sim_scene/run_feedback_sim.py > /experiment/sim/sim.log 2>&1'
