#!/bin/bash
set -euo pipefail
ROOT=/home/aluupy/arm_feedback_experiments/20261027
mkdir -p "$ROOT/diagnostics"
docker run --rm --network=none --entrypoint /isaac-sim/python.sh \
 -e LD_LIBRARY_PATH=/isaac-sim/extscache/omni.usd.libs-1.0.1+8131b85d.lx64.r.cp311/bin:/isaac-sim/kit/lib \
 -v /mnt/intel_ssd/omnilrs/OmniLRS/robots:/ws:ro -v "$ROOT:/experiment" \
 isaac-sim-omnilrs-0513:latest /experiment/src/tests/diagnose_static.py
docker exec arm-feedback-control-20261027 bash /experiment/src/sim_scene/run_python.sh tests/export_robot.py
