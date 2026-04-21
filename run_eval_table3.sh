#!/usr/bin/env bash
# run_eval_table3.sh
#
# Computes Table 3 (motion-transfer ablation) metrics using GT MRI videos
# from the dataset (same source policy as run_eval_table1.sh).
# No Stage-1 inference is run here.
#
# Ablation rows:
#   1) Preprocess=✗, Anchor=✗, Bidirectional=✗
#   2) Preprocess=✓, Anchor=✗, Bidirectional=✗
#   3) Preprocess=✓, Anchor=✓, Bidirectional=✗
#   4) Preprocess=✓, Anchor=✓, Bidirectional=✓
#
# Input layout:
#   ${DATASET_ROOT}/subXXX/2drt/video/subXXX_2drt_<clip>_video.(mp4|avi)
#
# Output:
#   ${OUTPUT_DIR}/table3_per_clip.csv
#   ${OUTPUT_DIR}/table3_summary.csv
#   ${OUTPUT_DIR}/table3_summary.txt
#   ${OUTPUT_DIR}/subXXX/<clip>/visualizations/<ablation>/*  (frame-by-frame images)
#
# Usage:
#   bash run_eval_table3.sh
#   MODEL_CONF="lstm_msessim_256_mri_melspectogram_baseline_ver0004_scene053_C_10" bash run_eval_table3.sh
#   SUBJECTS="002 009" CLIP_IDS="07_grandfather1_r1" bash run_eval_table3.sh

set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET_ROOT="${DATASET_ROOT:-/ssd1tb_00/dataset/mri_data}"
MODEL_CONF="${MODEL_CONF:-lstm_msessim_256_mri_melspectogram_baseline_ver0004_scene053}"
TABLE3_ROOT="${TABLE3_ROOT:-eval_output_table3}"
REF_MRI="${REF_MRI:-data_sample/ref_mri_0.png}"
REF_ANIME="${REF_ANIME:-data_sample/ref_anime_0.png}"
RAFT_MODEL="${RAFT_MODEL:-submodules/RAFT/models/raft-small.pth}"
DEVICE="${DEVICE:-cuda}"
MOTION_TAU="${MOTION_TAU:-0.5}"
RESUME="${RESUME:-0}"
SAVE_VIS="${SAVE_VIS:-1}"   # 1=save frame-by-frame visualizations
USE_SAVED_VIS="${USE_SAVED_VIS:-0}"  # 1=compute metrics from saved visualizations when available
PRE_SCALE_TARGET="${PRE_SCALE_TARGET:-1}"  # 1=upscale MRI frames to ref anime resolution (avoids cropped-looking outputs)

if [ -z "${SUBJECTS+x}" ]; then
    SUBJECTS="002 009 014 025 028 038 039 041 057 067"
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

OUTPUT_DIR="${OUTPUT_DIR:-${TABLE3_ROOT}/${MODEL_CONF}}"
mkdir -p "${OUTPUT_DIR}"
CLIP_IDS_JOINED="${CLIP_IDS[*]}"
export DATASET_ROOT OUTPUT_DIR REF_MRI REF_ANIME RAFT_MODEL DEVICE MOTION_TAU RESUME SUBJECTS CLIP_IDS_JOINED
export SAVE_VIS
export USE_SAVED_VIS
export PRE_SCALE_TARGET

echo "========================================================================"
echo "TABLE 3 Ablation Evaluation"
echo "  Dataset    : ${DATASET_ROOT}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Subjects   : ${SUBJECTS}"
echo "  Clips      : ${#CLIP_IDS[@]}"
echo "  Device     : ${DEVICE}"
echo "  Motion tau : ${MOTION_TAU}"
echo "  Save vis   : ${SAVE_VIS}"
echo "  Use saved  : ${USE_SAVED_VIS}"
echo "  Pre-scale  : ${PRE_SCALE_TARGET}"
echo "========================================================================"

python3 - <<'PYEOF'
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

project_root = Path.cwd()
sys.path.insert(0, str(project_root / "submodules" / "RAFT"))
sys.path.insert(0, str(project_root / "submodules" / "RAFT" / "core"))
from raft import RAFT  # type: ignore
from utils.flow_viz import flow_to_image

import importlib.util
metric_utils_spec = importlib.util.spec_from_file_location(
    "metric_utils", str(project_root / "utils" / "metric_utils.py")
)
if metric_utils_spec is None or metric_utils_spec.loader is None:
    raise RuntimeError("Failed to load utils/metric_utils.py")
metric_utils = importlib.util.module_from_spec(metric_utils_spec)
metric_utils_spec.loader.exec_module(metric_utils)

compute_stage3_metrics = metric_utils.compute_stage3_metrics


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


dataset_root = Path(env("DATASET_ROOT"))
output_dir = Path(env("OUTPUT_DIR"))
ref_mri_path = Path(env("REF_MRI"))
ref_anime_path = Path(env("REF_ANIME"))
raft_path = Path(env("RAFT_MODEL"))
device = env("DEVICE", "cuda")
motion_tau = float(env("MOTION_TAU", "0.5"))
resume = env("RESUME", "1") == "1"
save_vis = env("SAVE_VIS", "1") == "1"
use_saved_vis = env("USE_SAVED_VIS", "1") == "1"
pre_scale_target = env("PRE_SCALE_TARGET", "1") == "1"
subjects = [s for s in env("SUBJECTS").split() if s]
clip_ids = [c for c in env("CLIP_IDS_JOINED").split() if c]

if not dataset_root.is_dir():
    raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
if not ref_mri_path.is_file():
    raise FileNotFoundError(f"Reference MRI not found: {ref_mri_path}")
if not ref_anime_path.is_file():
    raise FileNotFoundError(f"Reference anime not found: {ref_anime_path}")
if not raft_path.is_absolute():
    raft_path = (project_root / raft_path).resolve()
if not raft_path.is_file():
    raise FileNotFoundError(f"RAFT checkpoint not found: {raft_path}")
if device == "cuda" and not torch.cuda.is_available():
    print("[WARN] CUDA unavailable, fallback to CPU.")
    device = "cpu"

output_dir.mkdir(parents=True, exist_ok=True)
per_clip_csv = output_dir / "table3_per_clip.csv"
summary_csv = output_dir / "table3_summary.csv"
summary_txt = output_dir / "table3_summary.txt"
detail_json = output_dir / "table3_detail.json"

if resume and detail_json.is_file():
    with detail_json.open() as f:
        cache = json.load(f)
else:
    cache = {}


def read_video_bgr(path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def resolve_gt_video(sub_id: str, clip_id: str) -> Optional[Path]:
    base = dataset_root / sub_id / "2drt" / "video"
    stem = f"{sub_id}_2drt_{clip_id}_video"
    for ext in (".mp4", ".avi"):
        p = base / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def vis_root_for(sub_id: str, clip_id: str, ablation: str) -> Path:
    return output_dir / sub_id / clip_id / "visualizations" / ablation


def has_saved_vis(sub_id: str, clip_id: str, ablation: str) -> bool:
    seq_dir = vis_root_for(sub_id, clip_id, ablation) / "temporal_sequence"
    return seq_dir.is_dir() and any(seq_dir.glob("anime_warped_*.png"))


def saved_vis_matches_target_shape(
    sub_id: str, clip_id: str, ablation: str, target_h: int, target_w: int
) -> bool:
    seq_dir = vis_root_for(sub_id, clip_id, ablation) / "temporal_sequence"
    anime0 = seq_dir / "anime_warped_0000.png"
    if not anime0.is_file():
        return False
    img = cv2.imread(str(anime0), cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    return (h, w) == (target_h, target_w)


def load_image_sequence(seq_dir: Path, prefix: str) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for p in sorted(seq_dir.glob(f"{prefix}*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    return frames


def get_registration_from_cache(cache: Dict, key: str) -> Tuple[int, float]:
    row = cache.get(key) if isinstance(cache, dict) else None
    if not isinstance(row, dict):
        return 0, float("nan")
    try:
        return int(row.get("anchor_index", 0)), float(row.get("registration_error", float("nan")))
    except Exception:
        return 0, float("nan")


def save_visualizations(
    sub_id: str,
    clip_id: str,
    ablation: str,
    ref_mri_gray: np.ndarray,
    ref_anime_bgr: np.ndarray,
    ref_anime_warped: np.ndarray,
    anchor_frame: np.ndarray,
    mri_frames: List[np.ndarray],
    anime_frames: List[np.ndarray],
    mri_flows: List[np.ndarray],
    anime_flows: List[np.ndarray],
) -> None:
    vis_root = vis_root_for(sub_id, clip_id, ablation)
    reg_dir = vis_root / "registration"
    seq_dir = vis_root / "temporal_sequence"
    flow_dir = vis_root / "optical_flows"
    reg_dir.mkdir(parents=True, exist_ok=True)
    seq_dir.mkdir(parents=True, exist_ok=True)
    flow_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(reg_dir / "ref_mri.png"), ref_mri_gray)
    cv2.imwrite(str(reg_dir / "ref_anime.png"), ref_anime_bgr)
    cv2.imwrite(str(reg_dir / "ref_anime_warped.png"), ref_anime_warped)
    cv2.imwrite(str(reg_dir / "anchor_mri_frame.png"), anchor_frame)
    h_mri, w_mri = ref_mri_gray.shape[:2]
    ref_mri_bgr = cv2.cvtColor(ref_mri_gray, cv2.COLOR_GRAY2BGR)
    anime_resized = cv2.resize(ref_anime_bgr, (w_mri, h_mri))
    warped_resized = cv2.resize(ref_anime_warped, (w_mri, h_mri))
    anchor_resized = cv2.resize(anchor_frame, (w_mri, h_mri))
    comp = np.hstack([ref_mri_bgr, anchor_resized, anime_resized, warped_resized])
    cv2.imwrite(str(reg_dir / "comparison.png"), comp)

    for i, frame in enumerate(mri_frames):
        cv2.imwrite(str(seq_dir / f"mri_{i:04d}.png"), frame)
    for i, frame in enumerate(anime_frames):
        cv2.imwrite(str(seq_dir / f"anime_warped_{i:04d}.png"), frame)

    for i in range(min(len(mri_flows), len(anime_flows))):
        mri_vis = flow_to_image(mri_flows[i], convert_to_bgr=True)
        anime_vis = flow_to_image(anime_flows[i], convert_to_bgr=True)
        cv2.imwrite(str(flow_dir / f"mri_flow_{i:04d}.png"), mri_vis)
        cv2.imwrite(str(flow_dir / f"anime_flow_{i:04d}.png"), anime_vis)
        h = max(mri_vis.shape[0], anime_vis.shape[0])
        w = max(mri_vis.shape[1], anime_vis.shape[1])
        mri_rs = cv2.resize(mri_vis, (w, h))
        anime_rs = cv2.resize(anime_vis, (w, h))
        cv2.imwrite(str(flow_dir / f"flow_comparison_{i:04d}.png"), np.hstack([mri_rs, anime_rs]))


def preprocess_pair(ref_gray: np.ndarray, target_gray: np.ndarray, enabled: bool) -> Tuple[np.ndarray, np.ndarray]:
    target_gray = cv2.resize(target_gray, (ref_gray.shape[1], ref_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    if not enabled:
        return ref_gray, target_gray
    ref_dn = cv2.medianBlur(ref_gray, 5)
    target_dn = cv2.medianBlur(target_gray, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(ref_dn), clahe.apply(target_dn)


def register_ecc(ref_img: np.ndarray, target_img: np.ndarray) -> Tuple[np.ndarray, float]:
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
    try:
        _, warp_matrix = cv2.findTransformECC(
            ref_img.astype(np.float32),
            target_img.astype(np.float32),
            warp_matrix,
            warp_mode,
            criteria,
        )
        h, w = ref_img.shape
        target_warped = cv2.warpAffine(
            target_img,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mse = float(np.mean((ref_img.astype(np.float32) - target_warped.astype(np.float32)) ** 2))
        return warp_matrix, mse
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32), float("inf")


def get_transform_and_anchor(
    ref_gray: np.ndarray,
    mri_gray_frames: List[np.ndarray],
    use_preprocess: bool,
    use_anchor: bool,
) -> Tuple[np.ndarray, int, float]:
    if not mri_gray_frames:
        return np.eye(2, 3, dtype=np.float32), 0, float("inf")

    if not use_anchor:
        ref_p, target_p = preprocess_pair(ref_gray, mri_gray_frames[0], use_preprocess)
        matrix, mse = register_ecc(ref_p, target_p)
        return matrix, 0, mse

    best_idx = 0
    best_mse = float("inf")
    best_matrix = np.eye(2, 3, dtype=np.float32)
    for i, frame in enumerate(mri_gray_frames):
        ref_p, target_p = preprocess_pair(ref_gray, frame, use_preprocess)
        matrix, mse = register_ecc(ref_p, target_p)
        if mse < best_mse:
            best_idx = i
            best_mse = mse
            best_matrix = matrix
    return best_matrix, best_idx, best_mse


def load_raft(path: Path, device_name: str) -> torch.nn.Module:
    model = RAFT(
        argparse.Namespace(
            model=str(path),
            small=True,
            mixed_precision=False,
            alternate_corr=False,
        )
    )
    ckpt = torch.load(str(path), map_location=device_name)
    if "module." in list(ckpt.keys())[0]:
        ckpt = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    return model.to(device_name).eval()


def flow_raft(
    frame1_bgr: np.ndarray,
    frame2_bgr: np.ndarray,
    raft_model: torch.nn.Module,
    device_name: str,
    flow_hw: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    h_orig, w_orig = frame1_bgr.shape[:2]
    tgt_h, tgt_w = flow_hw
    f1 = cv2.resize(frame1_bgr, (tgt_w, tgt_h))
    f2 = cv2.resize(frame2_bgr, (tgt_w, tgt_h))
    f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
    f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
    t1 = torch.from_numpy(f1).permute(2, 0, 1).float().unsqueeze(0).to(device_name)
    t2 = torch.from_numpy(f2).permute(2, 0, 1).float().unsqueeze(0).to(device_name)
    with torch.no_grad():
        _, flow_up = raft_model(t1, t2, iters=20, test_mode=True)
    flow = flow_up[0].cpu().numpy().transpose(1, 2, 0)
    flow = cv2.resize(flow, (w_orig, h_orig))
    flow[..., 0] *= w_orig / tgt_w
    flow[..., 1] *= h_orig / tgt_h
    return flow


def consecutive_flows(
    frames: List[np.ndarray], raft_model: torch.nn.Module, device_name: str
) -> List[np.ndarray]:
    return [flow_raft(frames[i], frames[i + 1], raft_model, device_name) for i in range(len(frames) - 1)]


def warp_anime_from_anchor(
    ref_anime_bgr: np.ndarray,
    rigid_transform: np.ndarray,
    flow_field: np.ndarray,
    target_dims: Tuple[int, int],
) -> np.ndarray:
    target_h, target_w = target_dims
    x, y = np.meshgrid(np.arange(target_w), np.arange(target_h))
    map_x_to_anchor = x - flow_field[:, :, 0]
    map_y_to_anchor = y - flow_field[:, :, 1]

    target_to_ref = cv2.invertAffineTransform(rigid_transform)
    coords_anchor = np.stack([map_x_to_anchor.flatten(), map_y_to_anchor.flatten()])
    coords_anchor_h = np.vstack([coords_anchor, np.ones(coords_anchor.shape[1])])
    coords_ref = target_to_ref @ coords_anchor_h

    final_map_x = coords_ref[0].reshape(target_h, target_w).astype(np.float32)
    final_map_y = coords_ref[1].reshape(target_h, target_w).astype(np.float32)
    return cv2.remap(
        ref_anime_bgr,
        final_map_x,
        final_map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def build_anime_sequence_anchor_relative(
    mri_frames_bgr: List[np.ndarray],
    ref_anime_bgr: np.ndarray,
    reg_matrix: np.ndarray,
    anchor_idx: int,
    bidirectional: bool,
    raft_model: torch.nn.Module,
    device_name: str,
    ref_mri_shape: Tuple[int, int],
) -> List[np.ndarray]:
    n = len(mri_frames_bgr)
    anchor_idx = max(0, min(anchor_idx, n - 1))
    h, w = mri_frames_bgr[0].shape[:2]

    rigid_transform = reg_matrix.copy()
    h_ref, w_ref = ref_mri_shape
    rigid_transform[0, :] *= (w / w_ref)
    rigid_transform[1, :] *= (h / h_ref)

    anchor_frame = mri_frames_bgr[anchor_idx]
    anchor_relative_flows = [flow_raft(anchor_frame, frame, raft_model, device_name) for frame in mri_frames_bgr]

    anime_frames: List[Optional[np.ndarray]] = [None] * n
    for t in range(anchor_idx, n):
        anime_frames[t] = warp_anime_from_anchor(
            ref_anime_bgr, rigid_transform, anchor_relative_flows[t], (h, w)
        )
    if bidirectional:
        for t in range(anchor_idx - 1, -1, -1):
            anime_frames[t] = warp_anime_from_anchor(
                ref_anime_bgr, rigid_transform, anchor_relative_flows[t], (h, w)
            )
    else:
        for t in range(anchor_idx - 1, -1, -1):
            anime_frames[t] = anime_frames[anchor_idx].copy()

    return [f for f in anime_frames if f is not None]


ablations = [
    ("none", False, False, False),
    ("preprocess_only", True, False, False),
    ("preprocess_anchor", True, True, False),
    ("full", True, True, True),
]

ref_mri_gray = cv2.imread(str(ref_mri_path), cv2.IMREAD_GRAYSCALE)
ref_anime_bgr = cv2.imread(str(ref_anime_path), cv2.IMREAD_COLOR)
if ref_mri_gray is None:
    raise RuntimeError(f"Failed to load {ref_mri_path}")
if ref_anime_bgr is None:
    raise RuntimeError(f"Failed to load {ref_anime_path}")

print(f"Loading RAFT model: {raft_path}")
raft = load_raft(raft_path, device)

rows = []
total = 0
done = 0
skip = 0
for sub in subjects:
    sub_id = f"sub{sub}"
    for clip in clip_ids:
        total += 1
        key_base = f"{sub_id}/{clip}"
        mri_video = resolve_gt_video(sub_id, clip)
        if mri_video is None:
            continue
        mri_frames = read_video_bgr(mri_video)
        if len(mri_frames) < 2:
            continue
        if pre_scale_target:
            target_h, target_w = ref_anime_bgr.shape[:2]
            if mri_frames[0].shape[:2] != (target_h, target_w):
                mri_frames = [
                    cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                    for frame in mri_frames
                ]
        mri_gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in mri_frames]
        mri_flows = consecutive_flows(mri_frames, raft, device)

        for ab_name, use_pre, use_anchor, use_bidir in ablations:
            cache_key = f"{key_base}::{ab_name}"
            vis_ready = (not save_vis) or (
                has_saved_vis(sub_id, clip, ab_name)
                and (
                    not pre_scale_target
                    or saved_vis_matches_target_shape(
                        sub_id, clip, ab_name, ref_anime_bgr.shape[0], ref_anime_bgr.shape[1]
                    )
                )
            )
            if resume and cache_key in cache and vis_ready:
                cached = cache[cache_key]
                # Cache version gate:
                # - dirsim must be normalized to [0,1]
                # - act_ratio must be present
                # - pre_scale_target mode must match
                if (
                    cached.get("dirsim_scale") == "0_1"
                    and "act_ratio" in cached
                    and bool(cached.get("pre_scale_target", False)) == pre_scale_target
                    and cached.get("warp_strategy") == "anchor_relative"
                    and cached.get("mri_source") == "gt_dataset"
                ):
                    rows.append(cached)
                    skip += 1
                    continue

            stage3 = None
            reg_matrix = None
            anchor_idx = 0
            reg_err = float("nan")
            anime_frames: List[np.ndarray] = []
            mri_frames_eval = mri_frames
            mri_flows_eval = mri_flows

            if use_saved_vis and has_saved_vis(sub_id, clip, ab_name):
                seq_dir = vis_root_for(sub_id, clip, ab_name) / "temporal_sequence"
                loaded_anime = load_image_sequence(seq_dir, "anime_warped_")
                loaded_mri = load_image_sequence(seq_dir, "mri_")
                if len(loaded_anime) >= 2:
                    if len(loaded_mri) >= 2:
                        n = min(len(loaded_mri), len(loaded_anime))
                        mri_frames_eval = loaded_mri[:n]
                        anime_frames = loaded_anime[:n]
                    else:
                        n = min(len(mri_frames), len(loaded_anime))
                        mri_frames_eval = mri_frames[:n]
                        anime_frames = loaded_anime[:n]
                    if len(mri_frames_eval) >= 2 and len(anime_frames) >= 2:
                        mri_flows_eval = consecutive_flows(mri_frames_eval, raft, device)
                        anime_flows = consecutive_flows(anime_frames, raft, device)
                        stage3 = compute_stage3_metrics(mri_flows_eval, anime_flows, tau=motion_tau)
                        anchor_idx, reg_err = get_registration_from_cache(cache, cache_key)

            if stage3 is None:
                mri_frames_eval = mri_frames
                mri_flows_eval = mri_flows
                reg_matrix, anchor_idx, reg_err = get_transform_and_anchor(
                    ref_mri_gray, mri_gray_frames, use_preprocess=use_pre, use_anchor=use_anchor
                )
                ref_anime_warped = cv2.warpAffine(
                    ref_anime_bgr,
                    reg_matrix,
                    (mri_frames[0].shape[1], mri_frames[0].shape[0]),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                anime_frames = build_anime_sequence_anchor_relative(
                    mri_frames_bgr=mri_frames,
                    ref_anime_bgr=ref_anime_bgr,
                    reg_matrix=reg_matrix,
                    anchor_idx=anchor_idx,
                    bidirectional=use_bidir,
                    raft_model=raft,
                    device_name=device,
                    ref_mri_shape=ref_mri_gray.shape,
                )
                if len(anime_frames) < 2:
                    continue
                anime_flows = consecutive_flows(anime_frames, raft, device)
                stage3 = compute_stage3_metrics(mri_flows_eval, anime_flows, tau=motion_tau)
            else:
                # placeholder for compatibility with save_visualizations path
                ref_anime_warped = (
                    cv2.warpAffine(
                        ref_anime_bgr,
                        reg_matrix,
                        (mri_frames[0].shape[1], mri_frames[0].shape[0]),
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                    if reg_matrix is not None
                    else anime_frames[0]
                )

            dirsim_raw = float(stage3.get("dirsim"))
            dirsim_01 = 0.5 * (dirsim_raw + 1.0)
            if save_vis and not (use_saved_vis and has_saved_vis(sub_id, clip, ab_name)):
                save_visualizations(
                    sub_id=sub_id,
                    clip_id=clip,
                    ablation=ab_name,
                    ref_mri_gray=ref_mri_gray,
                    ref_anime_bgr=ref_anime_bgr,
                    ref_anime_warped=ref_anime_warped,
                    anchor_frame=mri_frames[min(max(anchor_idx, 0), len(mri_frames) - 1)],
                    mri_frames=mri_frames_eval,
                    anime_frames=anime_frames,
                    mri_flows=mri_flows_eval,
                    anime_flows=anime_flows,
                )

            row = {
                "subject": sub_id,
                "clip": clip,
                "ablation": ab_name,
                "preprocess": int(use_pre),
                "anchor": int(use_anchor),
                "bidirectional": int(use_bidir),
                "anchor_index": int(anchor_idx),
                "registration_error": float(reg_err),
                "epe": float(stage3.get("epe")),
                "dirsim_raw": dirsim_raw,
                "dirsim": dirsim_01,
                "dirsim_scale": "0_1",
                "smooth": float(stage3.get("smooth")),
                "act_ratio": float(stage3.get("motion_activity_ratio")),
                "cov_ratio": float(stage3.get("motion_coverage_ratio")),
                "pre_scale_target": bool(pre_scale_target),
                "warp_strategy": "anchor_relative",
                "mri_source": "gt_dataset",
            }
            rows.append(row)
            cache[cache_key] = row
            done += 1
        print(f"[{sub_id}/{clip}] processed")

with detail_json.open("w") as f:
    json.dump(cache, f, indent=2)

if not rows:
    print("No rows computed.")
    sys.exit(0)

fieldnames = [
    "subject","clip","ablation","preprocess","anchor","bidirectional",
    "anchor_index","registration_error","epe","dirsim","dirsim_raw","dirsim_scale","smooth",
    "act_ratio","cov_ratio","pre_scale_target","warp_strategy","mri_source"
]
with per_clip_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

by_ablation: Dict[str, List[Dict]] = {}
for r in rows:
    by_ablation.setdefault(r["ablation"], []).append(r)

summary_rows = []
for ab_name, use_pre, use_anchor, use_bidir in ablations:
    vals = by_ablation.get(ab_name, [])
    if not vals:
        continue
    def m(k: str) -> float:
        return float(np.mean([v[k] for v in vals]))
    summary_rows.append({
        "ablation": ab_name,
        "preprocess": "✓" if use_pre else "",
        "anchor": "✓" if use_anchor else "",
        "bidirectional": "✓" if use_bidir else "",
        "n_clips": len(vals),
        "epe": m("epe"),
        "dirsim": m("dirsim"),
        "smooth": m("smooth"),
        "act_ratio": m("act_ratio"),
        "cov_ratio": m("cov_ratio"),
    })

with summary_csv.open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "ablation","preprocess","anchor","bidirectional","n_clips",
            "epe","dirsim","smooth","act_ratio","cov_ratio"
        ],
    )
    w.writeheader()
    w.writerows(summary_rows)

lines = []
header = "  {:<18} {:<10} {:<8} {:<13} {:>8} {:>10} {:>12} {:>10} {:>10} {:>10}".format(
    "Ablation", "Preprocess", "Anchor", "Bidirectional", "N", "EPE", "DirSim[0,1]", "Smooth", "ActRatio", "CovRatio"
)
sep = "  " + "-" * (len(header) - 2)
lines.extend([header, sep])
for r in summary_rows:
    lines.append(
        "  {:<18} {:<10} {:<8} {:<13} {:>8} {:>10.4f} {:>12.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r["ablation"], r["preprocess"], r["anchor"], r["bidirectional"],
            int(r["n_clips"]), r["epe"], r["dirsim"], r["smooth"], r["act_ratio"], r["cov_ratio"]
        )
    )
lines.append(sep)

with summary_txt.open("w") as f:
    f.write("TABLE 3 Motion-Transfer Ablation Summary\n\n")
    for line in lines:
        f.write(line + "\n")
    f.write("\nColumns map directly to paper Table 3.\n")

for line in lines:
    print(line)

print()
print(f"Processed combinations: {done}, resumed from cache: {skip}, total subject/clip slots: {total}")
print(f"Per-clip CSV : {per_clip_csv}")
print(f"Summary CSV  : {summary_csv}")
print(f"Summary TXT  : {summary_txt}")
PYEOF

echo "========================================================================"
echo "Done."
echo "========================================================================"
