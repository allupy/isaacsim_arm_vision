#!/bin/bash
# Stop only the two containers created by this experiment.
set -euo pipefail
for container in arm-feedback-control-20261027 arm-feedback-sim-20261027 arm-feedback-sim-diag-20261027 arm-feedback-sim-diag2-20261027; do
    if docker inspect "$container" >/dev/null 2>&1; then
        docker stop -t 10 "$container"
    fi
done
