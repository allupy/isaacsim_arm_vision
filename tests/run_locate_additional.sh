#!/bin/bash
set -euo pipefail
ROOT=/home/aluupy/arm_feedback_experiments/20261027
OUT="$ROOT/diagnostics/locate_feasibility_20261028"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3 MOONVIT_ATTN_IMPL=sdpa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PYTHON=/mnt/intel_ssd/conda_global/envs/aluupy/la/bin/python
for job in student06_head:scene student06_head:controls teacher_native:scene teacher_native:controls; do
  MODEL=${job%%:*}; DATA=${job##*:}; DIRECTORY="$OUT"
  if test "$DATA" = controls; then DIRECTORY="$OUT/controls"; fi
  "$PYTHON" -u "$ROOT/src/tests/locate_feasibility.py" --model "$MODEL" --out "$DIRECTORY" \
    > "$ROOT/locate_${MODEL}_${DATA}_20261028.log" 2>&1
done
