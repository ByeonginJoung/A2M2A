#!/usr/bin/env bash
# run_pipeline_video_eval.sh
#
# Runs the full A2M2A inference pipeline on a test video, then evaluates the
# generated outputs using the metrics defined in the IEEE Access 2026 paper:
#
#   Stage 1 – MRI Reconstruction  (MSE, PSNR, SSIM, Temporal Consistency)
#             Requires GT_VIDEO_FILE (the original MRI video from USC-TIMIT).
#             Set GT_VIDEO_FILE="" to skip Stage 1.
#
#   Stage 2 – Cross-Domain Registration  (Registration Error, Anchor Index)
#             Computed automatically during inference; saved to METRICS_JSON.
#
#   Stage 3 – Motion Faithfulness  (EPE, DirSim, Smooth)
#             Computed from RAFT optical flow on MRI and anime outputs.
#             Always computed when RAFT model is present.
#
# Usage:
#   bash run_pipeline_video_eval.sh
#
# Key environment variables you can override:
#   VIDEO_FILE     – input MRI video (USC-TIMIT .avi)
#   AUDIO_FILE     – paired audio (.wav); auto-detected from dataset if not set.
#                    Set to "" to fall back to extracting audio from VIDEO_FILE.
#   GT_VIDEO_FILE  – ground-truth MRI video for Stage 1; defaults to VIDEO_FILE.
#                    Set to "" to skip Stage 1.
#   DATASET_ROOT   – path to the USC-TIMIT dataset root
#   LOG_DIR        – path to trained model logs directory
#   CONF_NAME      – experiment / config name under LOG_DIR
#   OUTPUT_DIR     – directory for all outputs (MRI, anime, metrics)
# ────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ── video Input ────────────────────────────────────────
# USC-TIMIT MRI videos have no embedded audio stream; the corresponding
# _convert_ver2.wav file in the dataset directory must be provided separately.
DATASET_ROOT=${DATASET_ROOT:-/ssd1tb_00/dataset/mri_data}
VIDEO_FILE=${VIDEO_FILE:-demo_items/sub051_2drt_07_grandfather1_r1_video.avi}

# Auto-detect the paired audio file from the dataset when not set explicitly.
# Naming convention: <video_stem>_convert_ver2.wav alongside the video.
if [ -z "${AUDIO_FILE+x}" ]; then
    VIDEO_STEM=$(basename "${VIDEO_FILE%.*}")
    SUB_ID=$(echo "${VIDEO_STEM}" | grep -oP 'sub\d+' | head -1)
    CANDIDATE="${DATASET_ROOT}/${SUB_ID}/2drt/video/${VIDEO_STEM}_convert_ver2.wav"
    if [ -f "${CANDIDATE}" ]; then
        AUDIO_FILE="${CANDIDATE}"
        echo "Auto-detected audio: ${AUDIO_FILE}"
    else
        AUDIO_FILE=""
        echo "Warning: paired audio not found at ${CANDIDATE}; will attempt extraction from video."
    fi
fi

# Ground-truth MRI video for Stage 1 metrics (same file as the input when
# the input IS a raw MRI video from USC-TIMIT).
GT_VIDEO_FILE=${GT_VIDEO_FILE:-${VIDEO_FILE}}

# ── Model ──────────────────────
LOG_DIR=${LOG_DIR:-logs}
CONF_NAME=${CONF_NAME:-lstm_msessim_256_mri_melspectogram_baseline_ver0004_scene051}

# ── Outputs ───────────────────────────────────────────────────────────────────
OUTPUT_DIR=${OUTPUT_DIR:-eval_output}
METRICS_JSON=${OUTPUT_DIR}/registration_metrics.json
EVAL_JSON=${OUTPUT_DIR}/eval_metrics.json

# ──────
# Step 1 – Inference
# ─────────────────────────────────
echo "========================================================================"
echo "[Step 1] Running A2M2A inference..."
echo "  Video  : ${VIDEO_FILE}"
echo "  Audio  : ${AUDIO_FILE:-<extracted from video>}"
echo "  Model  : ${LOG_DIR}/${CONF_NAME}"
echo "  Output : ${OUTPUT_DIR}"
echo "========================================================================"

PIPELINE_ARGS=(
    --video_file "${VIDEO_FILE}"
    --output_dir "${OUTPUT_DIR}"
    --log_dir    "${LOG_DIR}/${CONF_NAME}"
    --save_metrics_path "${METRICS_JSON}"
    --concat_outputs
)

if [ -n "${AUDIO_FILE}" ] && [ -f "${AUDIO_FILE}" ]; then
    PIPELINE_ARGS+=(--audio_file "${AUDIO_FILE}")
fi

python run_pipeline_video.py "${PIPELINE_ARGS[@]}"

# 
# Step 2 – Locate generated videos
# ─────────────
PRED_MRI_VIDEO=$(ls "${OUTPUT_DIR}/mri/"*.mp4 2>/dev/null | head -1)
VIDEO_STEM=$(basename "${VIDEO_FILE%.*}")
PRED_ANIME_VIDEO="${OUTPUT_DIR}/anime/${VIDEO_STEM}.mp4"

if [ -z "${PRED_MRI_VIDEO}" ]; then
    echo "ERROR: No predicted MRI video found in ${OUTPUT_DIR}/mri/"
    exit 1
fi

if [ ! -f "${PRED_ANIME_VIDEO}" ]; then
    PRED_ANIME_VIDEO=$(ls "${OUTPUT_DIR}/anime/"*.mp4 2>/dev/null | head -1)
fi

if [ -z "${PRED_ANIME_VIDEO}" ]; then
    echo "ERROR: No predicted anime video found in ${OUTPUT_DIR}/anime/"
    exit 1
fi

echo ""
echo "  Predicted MRI  : ${PRED_MRI_VIDEO}"
echo "  Predicted Anime: ${PRED_ANIME_VIDEO}"

# ──────────────────────────────────────────────────────────────────
# Step 3 – Evaluation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "[Step 2] Running evaluation..."
echo "========================================================================"

EVAL_ARGS=(
    --pred_mri_video   "${PRED_MRI_VIDEO}"
    --pred_anime_video "${PRED_ANIME_VIDEO}"
    --output_json      "${EVAL_JSON}"
)

if [ -f "${METRICS_JSON}" ]; then
    EVAL_ARGS+=(--registration_metrics_json "${METRICS_JSON}")
fi

if [ -n "${GT_VIDEO_FILE}" ] && [ -f "${GT_VIDEO_FILE}" ]; then
    EVAL_ARGS+=(--gt_mri_video "${GT_VIDEO_FILE}")
else
    echo "  [Stage 1] Skipped — GT_VIDEO_FILE not set or file not found."
fi

python eval.py "${EVAL_ARGS[@]}"

echo ""
echo "========================================================================"
echo "Evaluation complete.  Results saved to: ${EVAL_JSON}"
echo "========================================================================"

# ──────────────────────────────────────────────────────────────────────────────
# Step 4 – Side-by-side concat (GT | Predicted MRI | Anime)
# ──────────────────────────────────────────────────────────────────────────────
CONCAT_OUTPUT="${OUTPUT_DIR}/concat_gt_mri_anime.mp4"
echo ""
echo "========================================================================"
echo "[Step 4] Creating side-by-side comparison video (GT | MRI | Anime)..."
echo "  Left   (GT)   : ${GT_VIDEO_FILE}"
echo "  Center (MRI)  : ${PRED_MRI_VIDEO}"
echo "  Right  (Anime): ${PRED_ANIME_VIDEO}"
echo "  Output        : ${CONCAT_OUTPUT}"
echo "========================================================================"

# Scale all three inputs to the same height (anime native: 456 px) then hstack.
# GT and predicted MRI are 84x84; they are upscaled proportionally.
ffmpeg -y \
    -i "${GT_VIDEO_FILE}" \
    -i "${PRED_MRI_VIDEO}" \
    -i "${PRED_ANIME_VIDEO}" \
    -filter_complex \
        "[0:v]scale=-2:456[v0];[1:v]scale=-2:456[v1];[2:v]scale=-2:456[v2];[v0][v1][v2]hstack=inputs=3[v]" \
    -map "[v]" \
    -c:v libx264 -crf 18 -preset fast \
    "${CONCAT_OUTPUT}"

echo ""
echo "Side-by-side video saved to: ${CONCAT_OUTPUT}"
