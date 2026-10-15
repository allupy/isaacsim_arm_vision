#!/bin/bash
# 启动(或重启)机械臂工作台仿真容器,直到稳定运行(带重试)
# 用法: bash start_sim.sh [duration_sec]
DUR=${1:-3600}
DBG=${DEBUG_RENDER:-0}
WS=/mnt/intel_ssd/omnilrs/OmniLRS/robots
LOG=$WS/_sim.log

for attempt in 1 2 3 4 5 6 7 8; do
    docker rm -f arm-wb-sim >/dev/null 2>&1
    sleep 2
    docker run -d --name arm-wb-sim --entrypoint bash \
        --gpus '"device=3"' --network=host --ipc=host \
        -e ROS_DOMAIN_ID=77 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
        -e SIM_DURATION=$DUR -e DEBUG_RENDER=$DBG \
        -v $HOME/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
        -v $HOME/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
        -v $HOME/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
        -v $HOME/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
        -v $WS:/ws -v /tmp/run_sim.py:/tmp/run_sim.py \
        isaac-sim-omnilrs-0513:latest \
        -c '/isaac-sim/python.sh /tmp/run_sim.py > /ws/_sim.log 2>&1'

    # 最多等 6 分钟判定是否稳定(sim running = 已持续 update 超过600帧)
    for i in $(seq 1 24); do
        sleep 15
        if grep -aq 'sim running' $LOG; then
            echo "STABLE (attempt $attempt, t+$((i*15))s)"
            exit 0
        fi
        if grep -aq 'Segmentation\|There was an error' $LOG; then
            echo "CRASHED (attempt $attempt, t+$((i*15))s), retrying..."
            break
        fi
    done
    grep -aq 'sim running' $LOG && exit 0
done
echo "FAILED after 8 attempts"
exit 1
