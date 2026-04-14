#!/bin/bash
#
# TABLE1 eval-only runner (precomputed visualizations).
# Does NOT generate MRI/anime. It only evaluates existing files under eval_output_table1.
#
# Usage:
#   bash run_eval_table1_precomputed.sh
#   METHOD_KEYS="Ours_Deep_Flow" SUBJECTS="002" CLIP_IDS="02_vcv2_r1" bash run_eval_table1_precomputed.sh
#   METHOD_KEYS="all" SUBJECTS="all" CLIP_IDS="all" bash run_eval_table1_precomputed.sh

set -euo pipefail

EVAL_ROOT="${EVAL_ROOT:-eval_output_table1}"
DATASET_ROOT="${DATASET_ROOT:-/ssd1tb_00/dataset/mri_data}"
SUBJECTS="${SUBJECTS:-all}"
CLIP_IDS="${CLIP_IDS:-all}"
METHOD_KEYS="${METHOD_KEYS:-all}"
DEVICE="${DEVICE:-cuda}"
MOTION_TAU="${MOTION_TAU:-0.5}"
NUM_FRAMES="${NUM_FRAMES:-}"
RESUME="${RESUME:-1}"  # 1=skip existing eval_metrics.json

ARGS=(
  --eval_root "${EVAL_ROOT}"
  --dataset_root "${DATASET_ROOT}"
  --subjects "${SUBJECTS}"
  --clip_ids "${CLIP_IDS}"
  --method_keys "${METHOD_KEYS}"
  --device "${DEVICE}"
  --motion_threshold "${MOTION_TAU}"
)

if [ -n "${NUM_FRAMES}" ]; then
  ARGS+=(--num_frames "${NUM_FRAMES}")
fi
if [ "${RESUME}" = "1" ]; then
  ARGS+=(--resume)
fi

echo "========================================================================"
echo "TABLE1 Precomputed Evaluation"
echo "  Eval Root:    ${EVAL_ROOT}"
echo "  Dataset Root: ${DATASET_ROOT}"
echo "  Subjects:     ${SUBJECTS}"
echo "  Clips:        ${CLIP_IDS}"
echo "  Methods:      ${METHOD_KEYS}"
echo "  Device:       ${DEVICE}"
echo "  Motion Tau:   ${MOTION_TAU}"
echo "========================================================================"

python eval_table1_precomputed.py "${ARGS[@]}"
