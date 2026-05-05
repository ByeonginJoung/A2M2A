#!/usr/bin/env bash
#
# TABLE3 eval-only runner (precomputed visualizations).
# Does NOT generate MRI/anime. It only evaluates existing files under eval_output_table3.
#
# Usage:
#   bash run_eval_table3_precomputed.sh
#   SUBJECTS="002 009" CLIP_IDS="01_vcv1_r1" ABLATIONS="full" bash run_eval_table3_precomputed.sh
#   SUBJECTS="all" CLIP_IDS="all" ABLATIONS="all" bash run_eval_table3_precomputed.sh

set -euo pipefail

TABLE3_ROOT="${TABLE3_ROOT:-eval_output_table3}"
MODEL_CONF="${MODEL_CONF:-lstm_msessim_256_mri_melspectogram_baseline_ver0004_scene053}"
EVAL_ROOT="${EVAL_ROOT:-${TABLE3_ROOT}/${MODEL_CONF}}"
DATASET_ROOT="${DATASET_ROOT:-/ssd1tb_00/dataset/mri_data}"
SUBJECTS="${SUBJECTS:-all}"
CLIP_IDS="${CLIP_IDS:-all}"
ABLATIONS="${ABLATIONS:-all}"
DEVICE="${DEVICE:-cuda}"
MOTION_TAU="${MOTION_TAU:-0.5}"
FLOW_H="${FLOW_H:-256}"
FLOW_W="${FLOW_W:-256}"
RESUME="${RESUME:-1}"  # 1=load rows from table3_detail_precomputed.json when compatible

ARGS=(
  --eval_root "${EVAL_ROOT}"
  --dataset_root "${DATASET_ROOT}"
  --subjects "${SUBJECTS}"
  --clip_ids "${CLIP_IDS}"
  --ablations "${ABLATIONS}"
  --device "${DEVICE}"
  --motion_threshold "${MOTION_TAU}"
  --flow_size "${FLOW_H}" "${FLOW_W}"
)

if [ "${RESUME}" = "1" ]; then
  ARGS+=(--resume)
fi

echo "========================================================================"
echo "TABLE3 Precomputed Evaluation"
echo "  Eval Root:    ${EVAL_ROOT}"
echo "  Dataset Root: ${DATASET_ROOT}"
echo "  Subjects:     ${SUBJECTS}"
echo "  Clips:        ${CLIP_IDS}"
echo "  Ablations:    ${ABLATIONS}"
echo "  Device:       ${DEVICE}"
echo "  Motion Tau:   ${MOTION_TAU}"
echo "  Flow Size:    ${FLOW_H}x${FLOW_W}"
echo "========================================================================"

python eval_table3_precomputed.py "${ARGS[@]}"

