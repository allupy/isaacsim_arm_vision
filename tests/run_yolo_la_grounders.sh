#!/bin/bash
set -euo pipefail
ROOT=/home/aluupy/arm_feedback_experiments/20261027
OUT="$ROOT/diagnostics/yolo_la_compare_20261028"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3 MOONVIT_ATTN_IMPL=sdpa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PYTHON=/mnt/intel_ssd/conda_global/envs/aluupy/la/bin/python
for MODEL in student06 teacher_native; do
  "$PYTHON" -u "$ROOT/src/tests/locate_feasibility.py" --model "$MODEL" --bbox-only --out "$OUT" \
    > "$ROOT/yolo_la_${MODEL}_20261028.log" 2>&1
done
