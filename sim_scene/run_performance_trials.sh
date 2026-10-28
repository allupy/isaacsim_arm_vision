#!/bin/bash
set -euo pipefail
ROOT=/home/aluupy/arm_feedback_experiments/20261027
MODE=${1:-optimized}
COUNT=${2:-3}
for i in $(seq 1 "$COUNT"); do
  RUN="performance_${MODE}_${i}"
  test ! -e "$ROOT/runs/$RUN"
  printf '%s' "$RUN" > "$ROOT/sim/reset_scene_requested"
  READY=0
  for attempt in $(seq 1 300); do
    if test -f "$ROOT/sim/reset_scene_ack" && test "$(cat "$ROOT/sim/reset_scene_ack")" = "$RUN"; then
      READY=1
      break
    fi
    sleep 1
  done
  test "$READY" = 1
  sleep 2
  ARGS=()
  if test "$MODE" = baseline; then ARGS+=(--legacy-performance); fi
  # Both profiles use the same conservative transfer ceiling for the controlled comparison.
  docker exec arm-feedback-control-20261027 bash /experiment/src/sim_scene/run_feedback.sh \
    --output "/experiment/runs/$RUN" --perception color --target-label red \
    --home .45 -.12 .5 --place-surface .4 -.3 .06 --tcp-offset .185 --safe-z .5 \
    "${ARGS[@]}" > "$ROOT/runs/$RUN.log" 2>&1 || true
  docker exec arm-feedback-control-20261027 cp /experiment/sim/command_updates.jsonl "/experiment/runs/$RUN/command_updates.jsonl"
  docker exec arm-feedback-control-20261027 cp /experiment/sim/performance.json "/experiment/runs/$RUN/simulator_performance.json"
  docker exec arm-feedback-control-20261027 cp /experiment/sim/execution.jsonl "/experiment/runs/$RUN/execution.jsonl"
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(sys.argv[1],r.get("success"),r.get("elapsed_s"),r.get("reason"),r.get("ground_truth_evaluation"))' "$ROOT/runs/$RUN/result.json"
done
docker exec arm-feedback-control-20261027 python3 /experiment/src/tests/summarize_performance.py /experiment/runs
