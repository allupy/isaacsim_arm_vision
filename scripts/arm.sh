#!/usr/bin/env bash
# Host-side launcher. Source stays read-only; outputs are per-user.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ARM_RUN_ROOT=${ARM_RUN_ROOT:-"$PROJECT_ROOT/.runtime/${USER:-user}"}
ROS_WS=${ROS_WS:-/home/aluupy/omnilrs_ros2_demo/humble_ws}
SCENE_DIR=${SCENE_DIR:-"$PROJECT_ROOT/scenes/workbench"}
SIM_IMAGE=${SIM_IMAGE:-isaac-sim-omnilrs-0513:latest}
CONTROL_IMAGE=${CONTROL_IMAGE:-moveit_aluupy:latest}
SIM_NAME=${SIM_NAME:-arm-${USER:-user}-sim}
CONTROL_NAME=${CONTROL_NAME:-arm-${USER:-user}-control}
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-78}
SIM_GPU=${SIM_GPU:-2}
CONTROL_GPU=${CONTROL_GPU:-3}
CACHE_ROOT=${CACHE_ROOT:-"$ARM_RUN_ROOT/cache"}
export ARM_RUN_ROOT CONTROL_NAME ROS_DOMAIN_ID

init() {
    mkdir -p "$ARM_RUN_ROOT"/{sim,runs,videos,agent_runs,diagnostics,assets} "$CACHE_ROOT"/{ov,glcache,computecache}
    if [[ ! -e "$ARM_RUN_ROOT/sim/runtime_config.json" ]]; then
        cp "$PROJECT_ROOT/sim_scene/runtime_everyday_demo.json" "$ARM_RUN_ROOT/sim/runtime_config.json"
    fi
}
check() {
    python3 "$PROJECT_ROOT/scripts/check_assets.py"
    test -f "$ROS_WS/moveit_ws/install/setup.bash" || { echo "Missing built MoveIt workspace: $ROS_WS" >&2; exit 1; }
    docker image inspect "$SIM_IMAGE" "$CONTROL_IMAGE" >/dev/null
}
start_sim() {
    init
    test -f "$SCENE_DIR/workbench.usd"
    if docker inspect "$SIM_NAME" >/dev/null 2>&1; then docker start "$SIM_NAME"; return; fi
    docker run -d --name "$SIM_NAME" --entrypoint bash \
        --gpus "\"device=$SIM_GPU\"" --network=host --ipc=private --shm-size=8g \
        -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e ROS_DOMAIN_ID \
        -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp -e FASTRTPS_DEFAULT_PROFILES_FILE=/tmp/udp_only.xml \
        -e SIM_DURATION="${SIM_DURATION:-3600}" -e CAM_EVERY=1 \
        -e REPAIR_PARKED_POSE=1 -e REPAIR_CAMERA_MOUNT=1 \
        -e PHYSICS_STEPS_PER_RENDER="${PHYSICS_STEPS_PER_RENDER:-120}" \
        -v "$ARM_RUN_ROOT:/experiment" -v "$PROJECT_ROOT:/experiment/src:ro" \
        -v "$PROJECT_ROOT/assets/ycb:/experiment/assets/ycb:ro" -v "$SCENE_DIR:/ws:ro" \
        -v "$CACHE_ROOT/ov:/root/.cache/ov" -v "$CACHE_ROOT/glcache:/root/.cache/nvidia/GLCache" \
        -v "$CACHE_ROOT/computecache:/root/.nv/ComputeCache" \
        -v "$PROJECT_ROOT/sim_scene/fastdds_udp.xml:/tmp/udp_only.xml:ro" \
        "$SIM_IMAGE" -c '/isaac-sim/python.sh /experiment/src/sim_scene/run_feedback_sim.py > /experiment/sim/sim.log 2>&1'
}
start_control() {
    init
    if docker inspect "$CONTROL_NAME" >/dev/null 2>&1; then docker start "$CONTROL_NAME"; return; fi
    docker run -d --name "$CONTROL_NAME" --entrypoint bash \
        --gpus "\"device=$CONTROL_GPU\"" --network=host --ipc=private \
        -e ROS_DOMAIN_ID -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp -e QT_QPA_PLATFORM=offscreen \
        -e FASTRTPS_DEFAULT_PROFILES_FILE=/experiment/src/sim_scene/fastdds_udp.xml \
        -e ROBOT_YOLO_WEIGHTS="${ROBOT_YOLO_WEIGHTS:-/ros2_ws/isaacsim_arm_vision/yolov8s-worldv2.pt}" \
        -e ROBOT_SAM_WEIGHTS="${ROBOT_SAM_WEIGHTS:-/ros2_ws/vision/sam_vit_b_01ec64.pth}" \
        -v "$ROS_WS:/ros2_ws:ro" -v "$ARM_RUN_ROOT:/experiment" \
        -v "$PROJECT_ROOT:/experiment/src:ro" \
        "$CONTROL_IMAGE" /experiment/src/sim_scene/control_services.sh
}
case "${1:-help}" in
    init) init ;;
    check) check ;;
    sim) start_sim ;;
    control) start_control ;;
    up) check; start_control; start_sim ;;
    stop)
        for name in "$CONTROL_NAME" "$SIM_NAME"; do
            if docker inspect "$name" >/dev/null 2>&1; then docker stop -t 10 "$name"; fi
        done ;;
    status) docker ps -a --filter "name=^/${SIM_NAME}$" --filter "name=^/${CONTROL_NAME}$" ;;
    agent) shift; python3 "$PROJECT_ROOT/agent_workflow/run.py" --root "$ARM_RUN_ROOT" "$@" ;;
    audit) shift; python3 "$PROJECT_ROOT/agent_workflow/audit.py" --root "$ARM_RUN_ROOT" "$@" ;;
    feedback) shift; docker exec "$CONTROL_NAME" bash /experiment/src/sim_scene/run_feedback.sh "$@" ;;
    interactive) shift; docker exec -it "$CONTROL_NAME" bash /experiment/src/sim_scene/run_python.sh main.py "$@" ;;
    *) echo 'Usage: bash scripts/arm.sh {init|check|up|sim|control|status|agent|audit|feedback|interactive|stop} [arguments]' ;;
esac
