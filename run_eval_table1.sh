#!/bin/bash

################################################################################
# TABLE 1 Baseline Evaluation Script (Batch Protocol)
#
# Evaluates 9 method combinations (6 baselines + 3 Ours variants)
# across multiple subjects and clips, following run_batch_eval.sh protocol.
#
# FAIR COMPARISON: All methods use GT MRI → anime (Stage 3 evaluation only)
#
# Subjects: 002 009 014 025 028 038 039 057 067 (configurable)
# Clips:    01_vcv1_r1, 02_vcv2_r1, ..., 10_northwind2_r1
#
# Usage:
#   bash run_eval_table1.sh                          # Default: all subjects
#   SUBJECTS="002 009" bash run_eval_table1.sh       # Custom subject list
#   NUM_FRAMES=50 bash run_eval_table1.sh            # Custom frame limit
#
# ── Configuration ──────────────────────────────────────────────────────────────
# Modify these variables to change evaluation settings:
#
DATASET_ROOT="${DATASET_ROOT:-/ssd1tb_00/dataset/mri_data}"
SUBJECTS="${SUBJECTS:-002 009 014 025 028 038 039 057 067}"  # Space-separated subject IDs
#SUBJECTS="${SUBJECTS:-}"  # Space-separated subject IDs
REF_ANIME="data_sample/ref_anime_0.png"
OUTPUT_DIR="${OUTPUT_DIR:-eval_output_table1}"
DEVICE="${DEVICE:-cuda}"
NUM_FRAMES="${NUM_FRAMES:-}"  # Empty = use all frames. Set to integer to limit (e.g., "100")
SAVE_VIS="${SAVE_VIS:-1}"  # Set to "1" to save qualitative visualizations for paper (DEFAULT: ENABLED)
METHOD_KEYS="${METHOD_KEYS:-Ours_Deep_Flow}"  # Optional CSV filter, e.g. "Ours_Deep_Flow" or "Ours+Deep Flow"
MOTION_TAU="${MOTION_TAU:-0.5}"  # Motion threshold tau for MotionActivityRatio

# Clips to evaluate (following run_batch_eval.sh protocol)
CLIP_IDS=(
    "01_vcv1_r1"
    "02_vcv2_r1"
    "03_vcv3_r1"
    "04_bvt_r1"
    "05_shibboleth_r1"
    "06_rainbow_r1"
    "07_grandfather1_r1"
    "08_grandfather2_r1"
    "09_northwind1_r1"
    "10_northwind2_r1"
)

################################################################################

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Validate configuration
if [ ! -f "$REF_ANIME" ]; then
    echo -e "${RED}Error: Reference anime image not found: $REF_ANIME${NC}"
    exit 1
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}TABLE 1 Baseline Evaluation (Batch Protocol)${NC}"
echo -e "${BLUE}FAIR COMPARISON: All methods use GT MRI → anime${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Configuration:"
echo "  Dataset Root:  $DATASET_ROOT"
echo "  Subjects:      $SUBJECTS"
echo "  Clips:         ${#CLIP_IDS[@]} (01_vcv1_r1, ..., 10_northwind2_r1)"
echo "  Output Dir:    $OUTPUT_DIR"
echo "  Device:        $DEVICE"
echo "  Frame Limit:   ${NUM_FRAMES:-all}"
echo "  Ref Anime:     $REF_ANIME"
echo "  Save Visuals:  ${SAVE_VIS} (1=yes, 0=no)"
echo "  Method Filter: ${METHOD_KEYS:-all}"
echo "  Motion Tau:    ${MOTION_TAU}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Initialize counters
TOTAL_COUNT=0
SKIP_COUNT=0
FAILED_COUNT=0
SUCCESS_COUNT=0

# Create temporary summary file
TEMP_SUMMARY="${OUTPUT_DIR}/temp_summary.json"
> "$TEMP_SUMMARY"  # Clear file

# Iterate over subjects and clips
for SUB in $SUBJECTS; do
    SUB_ID="sub${SUB}"
    VIDEO_DIR="${DATASET_ROOT}/${SUB_ID}/2drt/video"
    
    # Skip if subject directory doesn't exist
    if [ ! -d "$VIDEO_DIR" ]; then
        echo -e "${YELLOW}[SKIP]${NC} ${SUB_ID} — subject directory not found"
        continue
    fi
    
    for CLIP_ID in "${CLIP_IDS[@]}"; do
        VIDEO_FILE="${VIDEO_DIR}/${SUB_ID}_2drt_${CLIP_ID}_video.mp4"
        AUDIO_FILE="${VIDEO_DIR}/${SUB_ID}_2drt_${CLIP_ID}_video_convert_ver2.wav"
        CLIP_OUTPUT="${OUTPUT_DIR}/${SUB_ID}/${CLIP_ID}"
        EVAL_JSON="${CLIP_OUTPUT}/eval_metrics.json"
        
        TOTAL_COUNT=$((TOTAL_COUNT + 1))
        
        # Check if video exists
        if [ ! -f "$VIDEO_FILE" ]; then
            echo -e "${YELLOW}[SKIP]${NC} ${SUB_ID}/${CLIP_ID} — video not found"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi
        
        # Check if already evaluated for target method set
        ALL_METHODS=(
            "SuperPoint_RANSAC_Horn-Schunck"
            "SuperPoint_RANSAC_TV-L1"
            "SuperPoint_RANSAC_Deep_Flow"
            "LoFTR_Horn-Schunck"
            "LoFTR_TV-L1"
            "LoFTR_Deep_Flow"
            "Ours_Horn-Schunck"
            "Ours_TV-L1"
            "Ours_Deep_Flow"
        )

        TARGET_METHODS=()
        if [ -n "$METHOD_KEYS" ]; then
            IFS=',' read -ra TARGET_METHODS <<< "$METHOD_KEYS"
        else
            TARGET_METHODS=("${ALL_METHODS[@]}")
        fi
        
        ALL_EXIST=true
        MISSING_METHODS=()
        for METHOD in "${TARGET_METHODS[@]}"; do
            METHOD="$(echo "$METHOD" | xargs)"
            VIS_DIR="${CLIP_OUTPUT}/visualizations/${METHOD}"
            if [ ! -d "$VIS_DIR" ] || [ -z "$(ls -A "$VIS_DIR" 2>/dev/null)" ]; then
                ALL_EXIST=false
                MISSING_METHODS+=("$METHOD")
            fi
        done
        
        if [ -f "$EVAL_JSON" ] && [ "$ALL_EXIST" = true ]; then
            echo -e "${GREEN}[DONE]${NC} ${SUB_ID}/${CLIP_ID} — all ${#TARGET_METHODS[@]} target method(s) evaluated"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            continue
        elif [ -f "$EVAL_JSON" ]; then
            echo -e "${YELLOW}[PARTIAL]${NC} ${SUB_ID}/${CLIP_ID} — ${#MISSING_METHODS[@]} method(s) missing: ${MISSING_METHODS[*]}"
        fi
        
        mkdir -p "$CLIP_OUTPUT"
        
        echo -e "${BLUE}[EVAL]${NC} ${SUB_ID}/${CLIP_ID}"
        
        # Build Python command
        PYTHON_CMD="python eval_table1_baselines.py"
        PYTHON_CMD="$PYTHON_CMD --mri_video '$VIDEO_FILE'"
        PYTHON_CMD="$PYTHON_CMD --ref_anime '$REF_ANIME'"
        PYTHON_CMD="$PYTHON_CMD --output_json '$EVAL_JSON'"
        PYTHON_CMD="$PYTHON_CMD --device '$DEVICE'"
        
        # NOTE: "Ours" method now uses GT MRI (same as baselines) for fair comparison
        # No need for --log_dir or --gt_video_file anymore
        
        if [ -n "$NUM_FRAMES" ]; then
            PYTHON_CMD="$PYTHON_CMD --num_frames $NUM_FRAMES"
        fi
        
        if [ "$SAVE_VIS" = "1" ]; then
            PYTHON_CMD="$PYTHON_CMD --save_visualizations"
        fi
        
        if [ -n "$METHOD_KEYS" ]; then
            PYTHON_CMD="$PYTHON_CMD --method_keys '$METHOD_KEYS'"
        fi
        PYTHON_CMD="$PYTHON_CMD --motion_threshold '$MOTION_TAU'"
        
        # Run evaluation
        if eval "$PYTHON_CMD"; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo -e "${GREEN}  ✓ Success${NC}"
        else
            FAILED_COUNT=$((FAILED_COUNT + 1))
            echo -e "${RED}  ✗ Failed${NC}"
        fi
    done
done

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo "  Total:    $TOTAL_COUNT subject/clip pairs"
echo "  Success:  ${GREEN}$SUCCESS_COUNT${NC}"
echo "  Skipped:  ${YELLOW}$SKIP_COUNT${NC}"
echo "  Failed:   ${RED}$FAILED_COUNT${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Generate summary table
echo -e "${BLUE}Generating summary table...${NC}"
python3 << 'PYEOF'
import os
import json
import glob
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'eval_output_table1')

# Collect all metrics
all_results = {}
method_stats = defaultdict(lambda: {"count": 0, "epe": [], "dirsim": [], "smooth": [], "actratio": [], "covratio": []})

for json_file in glob.glob(f"{OUTPUT_DIR}/**/*/eval_metrics.json", recursive=True):
    try:
        with open(json_file) as f:
            data = json.load(f)
        
        sub_clip = json_file.replace(f"{OUTPUT_DIR}/", "").replace("/eval_metrics.json", "")
        all_results[sub_clip] = data
        
        # Aggregate by method
        for method_name, method_data in data.items():
            method_stats[method_name]["count"] += 1
            if "EPE" in method_data:
                method_stats[method_name]["epe"].append(method_data["EPE"])
            if "DirSim" in method_data:
                method_stats[method_name]["dirsim"].append(method_data["DirSim"])
            if "Smooth" in method_data:
                method_stats[method_name]["smooth"].append(method_data["Smooth"])
            if "MotionActivityRatio" in method_data:
                method_stats[method_name]["actratio"].append(method_data["MotionActivityRatio"])
            if "MotionCoverageRatio" in method_data:
                method_stats[method_name]["covratio"].append(method_data["MotionCoverageRatio"])
    except Exception as e:
        pass

if not all_results:
    print("No results found yet.")
    exit(0)

# Calculate means
print("\n" + "="*85)
print("METHOD COMPARISON (Aggregated across all subject/clip pairs)")
print("="*85)
print()
print(f"{'Method':<28} {'Count':>6} {'EPE':>10} {'DirSim':>10} {'Smooth':>10} {'ActRatio':>10} {'CovRatio':>10}")
print("-"*85)

for method in sorted(method_stats.keys()):
    stats = method_stats[method]
    count = stats["count"]
    epe = sum(stats["epe"]) / len(stats["epe"]) if stats["epe"] else None
    dirsim = sum(stats["dirsim"]) / len(stats["dirsim"]) if stats["dirsim"] else None
    smooth = sum(stats["smooth"]) / len(stats["smooth"]) if stats["smooth"] else None
    actratio = sum(stats["actratio"]) / len(stats["actratio"]) if stats["actratio"] else None
    covratio = sum(stats["covratio"]) / len(stats["covratio"]) if stats["covratio"] else None
    
    epe_str = f"{epe:.6f}" if epe is not None else "N/A"
    dirsim_str = f"{dirsim:.6f}" if dirsim is not None else "N/A"
    smooth_str = f"{smooth:.6f}" if smooth is not None else "N/A"
    actratio_str = f"{actratio:.6f}" if actratio is not None else "N/A"
    covratio_str = f"{covratio:.6f}" if covratio is not None else "N/A"
    
    print(f"{method:<28} {count:>6} {epe_str:>10} {dirsim_str:>10} {smooth_str:>10} {actratio_str:>10} {covratio_str:>10}")

print()
print(f"Total subject/clip pairs evaluated: {len(all_results)}")
print("="*85)
print()

PYEOF

echo -e "${GREEN}Done!${NC}"
