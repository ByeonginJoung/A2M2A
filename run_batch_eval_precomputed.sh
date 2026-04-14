#!/usr/bin/env bash
# run_batch_eval_precomputed.sh
#
# Re-evaluate existing outputs under eval_output_batch-style directories.
# This script does NOT run inference or anime generation.
#
# Expected structure:
#   ${BATCH_ROOT}/${MODEL_CONF}/${SUB_ID}/${CLIP_ID}/
#     ├── mri/   (predicted MRI video, e.g., output.avi or *.mp4)
#     ├── anime/ (predicted anime video, e.g., *.mp4)
#     └── registration_metrics.json (optional)
#
# Outputs:
#   ${BATCH_ROOT}/${MODEL_CONF}/${SUB_ID}/${CLIP_ID}/eval_metrics.json
#   ${BATCH_ROOT}/${MODEL_CONF}/summary.csv
#   ${BATCH_ROOT}/${MODEL_CONF}/ranking.txt
#   ${BATCH_ROOT}/model_comparison.csv
#   ${BATCH_ROOT}/model_comparison.txt

set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET_ROOT="${DATASET_ROOT:-/ssd1tb_00/dataset/mri_data}"
BATCH_ROOT="${BATCH_ROOT:-eval_output_batch}"
TOP_N="${TOP_N:-10}"
RESUME="${RESUME:-1}"                      # 1=skip clips with eval_metrics.json
DEVICE="${DEVICE:-cuda}"
MOTION_TAU="${MOTION_TAU:-0.5}"

if [ -z "${MODELS+x}" ]; then
    MODELS=(
        lstm_msessim_256_mri_melspectogram_baseline_ver0004_scene053_2
        lstm_msessim_256_mri_melspectogram_baseline_ver0004_scene053_6
        lstm_msessim_256_mri_melspectogram_baseline_ver0004_scene053
        lstm_msessim_256_mri_melspectogram_baseline_ver0004_scene053_A_2
        lstm_msessim_256_mri_melspectogram_baseline_ver0004_scene053_A_6
        lstm_msessim_256_mri_melspectogram_baseline_ver0004_scene053_A_10
        lstm_msessim_256_mri_melspectogram_baseline_ver0004_scene053_B_2
        lstm_msessim_256_mri_melspectogram_baseline_ver0004_scene053_B_6
        lstm_msessim_256_mri_melspectogram_baseline_ver0004_scene053_B_10
        lstm_msessim_256_mri_melspectogram_baseline_ver0004_scene053_C_2
        lstm_msessim_256_mri_melspectogram_baseline_ver0004_scene053_C_6
        lstm_msessim_256_mri_melspectogram_baseline_ver0004_scene053_C_10
    )
else
    read -ra MODELS <<< "${MODELS}"
fi

if [ -z "${CLIP_IDS+x}" ]; then
    CLIP_IDS=(
        01_vcv1_r1
        02_vcv2_r1
        03_vcv3_r1
        04_bvt_r1
        05_shibboleth_r1
        06_rainbow_r1
        07_grandfather1_r1
        08_grandfather2_r1
        09_northwind1_r1
        10_northwind2_r1
    )
else
    read -ra CLIP_IDS <<< "${CLIP_IDS}"
fi

if [ -n "${ALL_SUBJECTS+x}" ]; then
    SUBJECTS=$(seq -f '%03g' 1 75 | grep -v '^053$' | tr '\n' ' ')
elif [ -z "${SUBJECTS+x}" ]; then
    SUBJECTS="002 009 014 025 028 038 039 041 057 067"
fi

find_first_video() {
    local DIR="$1"
    local PREFERRED="$2"
    if [ -n "${PREFERRED}" ] && [ -f "${DIR}/${PREFERRED}" ]; then
        echo "${DIR}/${PREFERRED}"
        return 0
    fi
    local FOUND
    FOUND=$(ls "${DIR}/"*.mp4 "${DIR}/"*.avi "${DIR}/"*.mov 2>/dev/null | head -1 || true)
    if [ -n "${FOUND}" ]; then
        echo "${FOUND}"
        return 0
    fi
    return 1
}

run_model_eval() {
    local MODEL_CONF="$1"
    local MODEL_ROOT="${BATCH_ROOT}/${MODEL_CONF}"

    if [ ! -d "${MODEL_ROOT}" ]; then
        echo "[SKIP MODEL] Not found: ${MODEL_ROOT}"
        return 0
    fi

    echo ""
    echo "========================================================================"
    echo "  Precomputed Batch Evaluation"
    echo "  Model  : ${MODEL_CONF}"
    echo "  Root   : ${MODEL_ROOT}"
    echo "  Resume : ${RESUME}"
    echo "========================================================================"

    local DONE_COUNT=0
    local SKIP_COUNT=0
    local FAIL_COUNT=0

    for SUB in ${SUBJECTS}; do
        local SUB_ID="sub${SUB}"
        for CLIP_ID in "${CLIP_IDS[@]}"; do
            local CLIP_DIR="${MODEL_ROOT}/${SUB_ID}/${CLIP_ID}"
            local EVAL_JSON="${CLIP_DIR}/eval_metrics.json"
            local REG_JSON="${CLIP_DIR}/registration_metrics.json"
            local GT_VIDEO="${DATASET_ROOT}/${SUB_ID}/2drt/video/${SUB_ID}_2drt_${CLIP_ID}_video.mp4"

            if [ ! -d "${CLIP_DIR}" ]; then
                continue
            fi

            if [ "${RESUME}" = "1" ] && [ -f "${EVAL_JSON}" ]; then
                SKIP_COUNT=$((SKIP_COUNT + 1))
                continue
            fi

            local PRED_MRI_VIDEO
            local PRED_ANIME_VIDEO
            PRED_MRI_VIDEO=$(find_first_video "${CLIP_DIR}/mri" "output.avi" || true)
            PRED_ANIME_VIDEO=$(find_first_video "${CLIP_DIR}/anime" "" || true)

            if [ -z "${PRED_MRI_VIDEO}" ] || [ -z "${PRED_ANIME_VIDEO}" ]; then
                echo "[FAIL] ${SUB_ID}/${CLIP_ID} - missing mri/anime output video"
                FAIL_COUNT=$((FAIL_COUNT + 1))
                continue
            fi

            local ARGS=(
                --pred_mri_video "${PRED_MRI_VIDEO}"
                --pred_anime_video "${PRED_ANIME_VIDEO}"
                --output_json "${EVAL_JSON}"
                --device "${DEVICE}"
                --motion_threshold "${MOTION_TAU}"
            )

            if [ -f "${GT_VIDEO}" ]; then
                ARGS+=(--gt_mri_video "${GT_VIDEO}")
            fi
            if [ -f "${REG_JSON}" ]; then
                ARGS+=(--registration_metrics_json "${REG_JSON}")
            fi

            if python eval_precomputed.py "${ARGS[@]}"; then
                DONE_COUNT=$((DONE_COUNT + 1))
            else
                echo "[FAIL] ${SUB_ID}/${CLIP_ID} - evaluation error"
                FAIL_COUNT=$((FAIL_COUNT + 1))
            fi
        done
    done

    echo ""
    echo "  Aggregating results for ${MODEL_CONF} ..."

    MODEL_ROOT="${MODEL_ROOT}" TOP_N="${TOP_N}" python3 - <<'PYEOF'
import csv, json, os, sys

model_root = os.environ["MODEL_ROOT"]
top_n = int(os.environ.get("TOP_N", "10"))
summary_csv = os.path.join(model_root, "summary.csv")
ranking_txt = os.path.join(model_root, "ranking.txt")

rows = []
for sub_dir in sorted(os.listdir(model_root)):
    sub_path = os.path.join(model_root, sub_dir)
    if not os.path.isdir(sub_path) or not sub_dir.startswith("sub"):
        continue
    for clip_dir in sorted(os.listdir(sub_path)):
        clip_path = os.path.join(sub_path, clip_dir)
        if not os.path.isdir(clip_path):
            continue
        json_path = os.path.join(clip_path, "eval_metrics.json")
        if not os.path.isfile(json_path):
            continue
        try:
            with open(json_path) as f:
                m = json.load(f)
        except Exception:
            continue
        s1 = m.get("stage1", {})
        s2 = m.get("stage2", {})
        s3 = m.get("stage3", {})
        rows.append({
            "subject": sub_dir,
            "clip": clip_dir,
            "mse": s1.get("mse"),
            "psnr": s1.get("psnr"),
            "ssim": s1.get("ssim"),
            "ltemp": s1.get("temporal_consistency"),
            "reg_error": s2.get("registration_error"),
            "anchor_index": s2.get("anchor_index"),
            "epe": s3.get("epe"),
            "dir_sim": s3.get("dirsim"),
            "smooth": s3.get("smooth"),
            "motion_activity_ratio": s3.get("motion_activity_ratio"),
            "motion_coverage_ratio": s3.get("motion_coverage_ratio"),
        })

if not rows:
    print("  No results found.")
    sys.exit(0)

fieldnames = [
    "subject","clip","mse","psnr","ssim","ltemp","reg_error","anchor_index",
    "epe","dir_sim","smooth","motion_activity_ratio","motion_coverage_ratio"
]
with open(summary_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print(f"  Summary CSV : {summary_csv}  ({len(rows)} rows)")

def vals(key):
    return [r[key] for r in rows if r.get(key) is not None]
def mean(key):
    v = vals(key)
    return (sum(v)/len(v)) if v else None
def fmt(v, d=4):
    return f"{v:.{d}f}" if v is not None else "N/A"

ranked = [r for r in rows if r.get("epe") is not None]
ranked.sort(key=lambda r: r["epe"])

lines = []
lines.append("  {:<10} {:<22} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}".format(
    "Subject","Clip","EPE","DirSim","Smooth","PSNR","ActRatio","CovRatio"
))
lines.append("  " + "-"*90)
for r in ranked[:top_n]:
    lines.append("  {:<10} {:<22} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}".format(
        r["subject"], r["clip"], fmt(r["epe"]), fmt(r["dir_sim"]), fmt(r["smooth"]),
        fmt(r["psnr"],2), fmt(r["motion_activity_ratio"]), fmt(r["motion_coverage_ratio"])
    ))
lines.append("  " + "-"*90)
lines.append("  {:<10} {:<22} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}".format(
    "MEAN", f"({len(rows)} clips)",
    fmt(mean("epe")), fmt(mean("dir_sim")), fmt(mean("smooth")), fmt(mean("psnr"),2),
    fmt(mean("motion_activity_ratio")), fmt(mean("motion_coverage_ratio"))
))

for l in lines:
    print(l)

with open(ranking_txt, "w") as f:
    f.write("Ranking by: EPE (lower is better)\n")
    f.write(f"Total evaluated: {len(rows)} subject/clip pairs\n\n")
    for l in lines:
        f.write(l + "\n")
    f.write(f"\nFull results: {summary_csv}\n")
print(f"  Ranking txt : {ranking_txt}")
PYEOF

    echo ""
    echo "  Done: ${DONE_COUNT} new  |  Skipped: ${SKIP_COUNT}  |  Failed: ${FAIL_COUNT}"
    echo "========================================================================"
}

echo "========================================================================"
echo "  A2M2A Precomputed Multi-Model Evaluation"
echo "  Models  : ${#MODELS[@]}"
echo "  Subjects: ${SUBJECTS}"
echo "  Clips   : ${#CLIP_IDS[@]}"
echo "  Root    : ${BATCH_ROOT}"
echo "========================================================================"

for MODEL_CONF in "${MODELS[@]}"; do
    run_model_eval "${MODEL_CONF}"
done

echo ""
echo "========================================================================"
echo "  Cross-model comparison (mean metrics)"
echo "========================================================================"

BATCH_ROOT="${BATCH_ROOT}" python3 - <<'PYEOF'
import csv, os, sys

batch_root = os.environ["BATCH_ROOT"]
comp_csv = os.path.join(batch_root, "model_comparison.csv")
comp_txt = os.path.join(batch_root, "model_comparison.txt")

def col_mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals)/len(vals) if vals else None
def fmt(v, d=4):
    return f"{v:.{d}f}" if v is not None else "N/A"

model_rows = []
for model_dir in sorted(os.listdir(batch_root)):
    model_path = os.path.join(batch_root, model_dir)
    summary_csv = os.path.join(model_path, "summary.csv")
    if not os.path.isdir(model_path) or not os.path.isfile(summary_csv):
        continue
    rows = []
    with open(summary_csv) as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v) if v not in ("", "None", "N/A") else None
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    if not rows:
        continue
    model_rows.append({
        "model": model_dir,
        "n_clips": len(rows),
        "psnr": col_mean(rows, "psnr"),
        "mse": col_mean(rows, "mse"),
        "ssim": col_mean(rows, "ssim"),
        "ltemp": col_mean(rows, "ltemp"),
        "reg_error": col_mean(rows, "reg_error"),
        "epe": col_mean(rows, "epe"),
        "dir_sim": col_mean(rows, "dir_sim"),
        "smooth": col_mean(rows, "smooth"),
        "motion_activity_ratio": col_mean(rows, "motion_activity_ratio"),
        "motion_coverage_ratio": col_mean(rows, "motion_coverage_ratio"),
    })

if not model_rows:
    print("  No model summaries found.")
    sys.exit(0)

model_rows.sort(key=lambda r: (r["epe"] is None, (r["epe"] or 0.0)))

fieldnames = [
    "model","n_clips","psnr","mse","ssim","ltemp","reg_error","epe","dir_sim","smooth",
    "motion_activity_ratio","motion_coverage_ratio"
]
with open(comp_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(model_rows)

max_name = max(len(r["model"]) for r in model_rows)
wname = max(10, max_name)
header = "  {:<{w}} {:>7} {:>8} {:>8} {:>8} {:>10} {:>10} {:>8} {:>8} {:>10} {:>10}".format(
    "Model","N","EPE","DirSim","Smooth","PSNR","RegErr","Act","Cov","MSE","SSIM",w=wname
)
sep = "  " + "-"*(len(header)-2)
lines = [header, sep]
for r in model_rows:
    lines.append("  {:<{w}} {:>7} {:>8} {:>8} {:>8} {:>10} {:>10} {:>8} {:>8} {:>10} {:>10}".format(
        r["model"], int(r["n_clips"]),
        fmt(r["epe"]), fmt(r["dir_sim"]), fmt(r["smooth"]),
        fmt(r["psnr"],2), fmt(r["reg_error"]), fmt(r["motion_activity_ratio"]),
        fmt(r["motion_coverage_ratio"]), fmt(r["mse"]), fmt(r["ssim"]), w=wname
    ))
lines.append(sep)
for l in lines:
    print(l)

with open(comp_txt, "w") as f:
    f.write("Cross-model comparison (mean metrics)\n\n")
    for l in lines:
        f.write(l + "\n")
    f.write(f"\nFull results: {comp_csv}\n")
print(f"\n  Comparison CSV : {comp_csv}")
print(f"  Comparison txt : {comp_txt}")
PYEOF

echo ""
echo "========================================================================"
echo "  Done. Results under: ${BATCH_ROOT}/"
echo "========================================================================"
