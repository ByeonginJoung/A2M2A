#!/usr/bin/env python3
"""
Evaluate already-generated MRI/Anime videos (no generation/deformation).

Output JSON schema matches run_batch_eval.sh usage:
{
  "stage1": {...},   # optional when --gt_mri_video is provided
  "stage2": {...},   # optional when --registration_metrics_json exists
  "stage3": {...}    # always computed
}
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "submodules" / "RAFT"))
sys.path.insert(0, str(PROJECT_ROOT / "submodules" / "RAFT" / "core"))

from raft import RAFT  # type: ignore

_metric_utils_spec = importlib.util.spec_from_file_location(
    "metric_utils", str(PROJECT_ROOT / "utils" / "metric_utils.py")
)
if _metric_utils_spec is None or _metric_utils_spec.loader is None:
    raise RuntimeError("Failed to load utils/metric_utils.py")
_metric_utils = importlib.util.module_from_spec(_metric_utils_spec)
_metric_utils_spec.loader.exec_module(_metric_utils)
compute_stage1_metrics = _metric_utils.compute_stage1_metrics
compute_stage3_metrics = _metric_utils.compute_stage3_metrics


def load_video_bgr(video_path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def load_video_gray(video_path: Path, resize: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
    frames_bgr = load_video_bgr(video_path)
    out: List[np.ndarray] = []
    for frame in frames_bgr:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if resize is not None:
            gray = cv2.resize(gray, resize, interpolation=cv2.INTER_LINEAR)
        out.append(gray)
    return out


def get_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps


def temporal_downsample(frames: Sequence[np.ndarray], src_fps: float, tgt_fps: float) -> List[np.ndarray]:
    if src_fps <= 0 or tgt_fps <= 0 or abs(src_fps - tgt_fps) < 1e-6:
        return list(frames)
    ratio = src_fps / tgt_fps
    n_out = max(1, int(len(frames) / ratio))
    idx = [int(round(i * ratio)) for i in range(n_out)]
    idx = [i for i in idx if i < len(frames)]
    return [frames[i] for i in idx]


def compute_flow_raft(
    frame1_bgr: np.ndarray,
    frame2_bgr: np.ndarray,
    raft_model: torch.nn.Module,
    device: str,
    flow_hw: Tuple[int, int],
) -> np.ndarray:
    h_orig, w_orig = frame1_bgr.shape[:2]
    tgt_h, tgt_w = flow_hw

    f1 = cv2.resize(frame1_bgr, (tgt_w, tgt_h))
    f2 = cv2.resize(frame2_bgr, (tgt_w, tgt_h))
    f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
    f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)

    t1 = torch.from_numpy(f1).permute(2, 0, 1).float().unsqueeze(0).to(device)
    t2 = torch.from_numpy(f2).permute(2, 0, 1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        _, flow_up = raft_model(t1, t2, iters=20, test_mode=True)

    flow = flow_up[0].cpu().numpy().transpose(1, 2, 0)
    flow = cv2.resize(flow, (w_orig, h_orig))
    flow[..., 0] *= w_orig / tgt_w
    flow[..., 1] *= h_orig / tgt_h
    return flow


def compute_consecutive_flows(
    frames_bgr: Sequence[np.ndarray],
    raft_model: torch.nn.Module,
    device: str,
    flow_hw: Tuple[int, int],
) -> List[np.ndarray]:
    flows: List[np.ndarray] = []
    for i in range(len(frames_bgr) - 1):
        flows.append(compute_flow_raft(frames_bgr[i], frames_bgr[i + 1], raft_model, device, flow_hw))
    return flows


def load_raft(ckpt: Path, device: str) -> torch.nn.Module:
    model = RAFT(
        argparse.Namespace(
            model=str(ckpt),
            small=True,
            mixed_precision=False,
            alternate_corr=False,
        )
    )
    state = torch.load(str(ckpt), map_location=device)
    if "module." in list(state.keys())[0]:
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pre-generated MRI/Anime videos.")
    parser.add_argument("--pred_mri_video", required=True)
    parser.add_argument("--pred_anime_video", required=True)
    parser.add_argument("--gt_mri_video", default=None)
    parser.add_argument("--registration_metrics_json", default=None)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--raft_model", default="submodules/RAFT/models/raft-small.pth")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--flow_size", nargs=2, type=int, default=[256, 256], metavar=("H", "W"))
    parser.add_argument("--motion_threshold", type=float, default=0.5)
    args = parser.parse_args()

    pred_mri = Path(args.pred_mri_video)
    pred_anime = Path(args.pred_anime_video)
    gt_mri = Path(args.gt_mri_video) if args.gt_mri_video else None
    reg_json = Path(args.registration_metrics_json) if args.registration_metrics_json else None
    out_json = Path(args.output_json)

    if not pred_mri.is_file():
        raise FileNotFoundError(f"Missing pred_mri_video: {pred_mri}")
    if not pred_anime.is_file():
        raise FileNotFoundError(f"Missing pred_anime_video: {pred_anime}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, fallback to CPU.")
        device = "cpu"

    raft_model_path = Path(args.raft_model)
    if not raft_model_path.is_absolute():
        raft_model_path = (PROJECT_ROOT / raft_model_path).resolve()
    if not raft_model_path.is_file():
        raise FileNotFoundError(f"Missing RAFT checkpoint: {raft_model_path}")

    results = {}

    if gt_mri and gt_mri.is_file():
        gt_fps = get_video_fps(gt_mri)
        pred_fps = get_video_fps(pred_mri)
        gt_frames = load_video_gray(gt_mri)
        if gt_fps > 0 and pred_fps > 0 and abs(gt_fps - pred_fps) > 0.5:
            gt_frames = temporal_downsample(gt_frames, gt_fps, pred_fps)

        pred_frames = load_video_gray(pred_mri)
        if gt_frames and pred_frames:
            gh, gw = gt_frames[0].shape
            pred_frames = [
                cv2.resize(frame, (gw, gh), interpolation=cv2.INTER_LINEAR) for frame in pred_frames
            ]
            results["stage1"] = compute_stage1_metrics(gt_frames, pred_frames)

    if reg_json and reg_json.is_file():
        try:
            results["stage2"] = json.loads(reg_json.read_text())
        except Exception:
            pass

    raft = load_raft(raft_model_path, device)
    flow_hw = (args.flow_size[0], args.flow_size[1])

    mri_frames = load_video_bgr(pred_mri)
    anime_frames = load_video_bgr(pred_anime)
    if len(mri_frames) < 2 or len(anime_frames) < 2:
        raise RuntimeError(
            f"Need >=2 frames in each video (MRI={len(mri_frames)}, Anime={len(anime_frames)})."
        )

    mri_flows = compute_consecutive_flows(mri_frames, raft, device, flow_hw)
    anime_flows = compute_consecutive_flows(anime_frames, raft, device, flow_hw)
    stage3 = compute_stage3_metrics(mri_flows, anime_flows, tau=args.motion_threshold)
    stage3["flow_method"] = "deep_flow"
    results["stage3"] = stage3

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Saved metrics: {out_json}")
    print(
        f"Stage3 -> EPE={stage3['epe']:.6f}, DirSim={stage3['dirsim']:.6f}, "
        f"Smooth={stage3['smooth']:.6f}"
    )


if __name__ == "__main__":
    main()
