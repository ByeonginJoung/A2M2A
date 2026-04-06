#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Where to discover subject folders (used when SUB_NAMES is not provided).
DATASET_ROOT="${DATASET_ROOT:-/ssd1tb_00/dataset/mri_data}"

# Allow manual control over the subject ids (space-separated list). When empty we discover
# them automatically from DATASET_ROOT (looking for directories named 'subXXX').
if [[ -n "${SUB_NAMES:-}" ]]; then
    read -r -a SUB_NAME_ARRAY <<< "$SUB_NAMES"
else
    mapfile -t SUB_NAME_ARRAY < <(
        python - <<'PY'
import os
root = os.environ.get('DATASET_ROOT', '/ssd1tb_00/dataset/mri_data')
subs = []
for name in os.listdir(root):
    if name.startswith('sub') and len(name) >= 4:
        suffix = name[3:]
        if suffix.isdigit():
            subs.append(suffix)
if not subs:
    raise SystemExit('No subject folders found under %s' % root)
for sub in sorted(subs):
    print(sub)
PY
    ) || {
        echo "[run_train.sh] Failed to discover subject IDs under ${DATASET_ROOT}. Set SUB_NAMES manually." >&2
        exit 1
    }
fi

if [[ ${#SUB_NAME_ARRAY[@]} -eq 0 ]]; then
    echo "[run_train.sh] No subject IDs provided or discovered." >&2
    exit 1
fi

# Template strings for config/experiment names. {SUB} expands to the subject id, {CONF}
# expands to the resolved config name for that subject.
CONF_TEMPLATE="${CONF_TEMPLATE:-mri_melspectogram_baseline_ver0004_scene{SUB}}"
EXP_TEMPLATE="${EXP_TEMPLATE:-lstm_msessim_256_{CONF}}"

DATASET_TYPE="${DATASET_TYPE:-75-speaker-multi}"  # can be overridden per data variant

for SUB_NAME in "${SUB_NAME_ARRAY[@]}"; do
    CONF_NAME=${CONF_TEMPLATE//\{SUB\}/$SUB_NAME}
    EXP_NAME=${EXP_TEMPLATE//\{SUB\}/$SUB_NAME}
    EXP_NAME=${EXP_NAME//\{CONF\}/$CONF_NAME}

    echo "[run_train.sh] Training subject ${SUB_NAME} using config ${CONF_NAME} -> experiment ${EXP_NAME}"

    cmd=(
        "$PYTHON_BIN" train.py
        --dataset mri
        --exp_name "$EXP_NAME"
        --sub_name "$SUB_NAME"
        --config_name "$CONF_NAME"
        --dataset_type "$DATASET_TYPE"
    )

    "${cmd[@]}" "$@"
done
