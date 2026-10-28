#!/bin/bash
set -euo pipefail
ROOT=/home/aluupy/arm_feedback_experiments/20261027
MODE=${1:-original}
UNITS=${2:-.01}
NAME=arm-physics-probe-20261027
docker run --rm --name "$NAME" --entrypoint bash --gpus '"device=2"' \
 --network=none --ipc=private --shm-size=4g -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
 -e PROBE_MODE="$MODE" -e PROBE_UNITS="$UNITS" \
 -v /mnt/intel_ssd/omnilrs/OmniLRS/robots:/ws:ro -v "$ROOT:/experiment" \
 -v /home/aluupy/docker/isaac-sim/cache_wb/ov:/root/.cache/ov \
 -v /home/aluupy/docker/isaac-sim/cache_wb/glcache:/root/.cache/nvidia/GLCache \
 isaac-sim-omnilrs-0513:latest \
 -c '/isaac-sim/python.sh /experiment/src/tests/physics_probe.py' \
 > "$ROOT/diagnostics/probe_${MODE}_${UNITS}.log" 2>&1
