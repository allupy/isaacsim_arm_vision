#!/bin/bash
# 启动(或重启)机械臂工作台仿真容器,直到稳定运行(带重试)
# 用法: bash start_sim.sh [duration_sec]   GPU 用 SIM_GPU=N 覆盖(默认 1)
# 关键: 必须 --ipc=private!共享宿主 /dev/shm 池含大量陈旧 carb 段(死PID),
#       kit 启动时撞上会导致堆损坏→随机位置段错误/卡死(2026-10-16 排障结论)。
#       DDS 走 udp_only,与宿主其它 FastDDS 参与者隔离。
DUR=${1:-3600}
DBG=${DEBUG_RENDER:-0}
GPU=${SIM_GPU:-1}
WS=/mnt/intel_ssd/omnilrs/OmniLRS/robots
LOG=$WS/_sim.log
C=/home/aluupy/docker/isaac-sim/cache_wb
SIMDIR=$HOME/omnilrs_ros2_demo/humble_ws/isaacsim_arm_vision/sim_scene

for attempt in 1 2 3 4 5 6 7 8; do
    docker rm -f arm-wb-sim >/dev/null 2>&1
    sleep 2
    docker run -d --name arm-wb-sim --entrypoint bash         --gpus "\"device=$GPU\"" --network=host --ipc=private --shm-size=8g         -e ROS_DOMAIN_ID=77 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp         -e FASTRTPS_DEFAULT_PROFILES_FILE=/tmp/udp_only.xml         -e SIM_DURATION=$DUR -e DEBUG_RENDER=$DBG         -v $C/kit:/isaac-sim/kit/cache:rw         -v $C/kitdata:/isaac-sim/kit/data:rw         -v $C/ov:/root/.cache/ov:rw         -v $C/glcache:/root/.cache/nvidia/GLCache:rw         -v $C/computecache:/root/.nv/ComputeCache:rw         -v $WS:/ws -v $SIMDIR/run_sim.py:/tmp/run_sim.py         -v $SIMDIR/fastdds_udp.xml:/tmp/udp_only.xml:ro         isaac-sim-omnilrs-0513:latest         -c '/isaac-sim/python.sh /tmp/run_sim.py > /ws/_sim.log 2>&1'
    for i in $(seq 1 40); do
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
