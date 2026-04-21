#!/usr/bin/env python3
"""
Evaluation script for the A2M2A pipeline.

Implements the three groups of metrics described in the paper
"Audio-Driven Articulatory 2D Animation Synthesis via MRI Generation
and Cross-Domain Motion Transfer" (IEEE Access 2026).

Metric groups
─────────────
Stage 1 – MRI Reconstruction (requires --gt_mri_video):
    MSE, PSNR, SSIM, Temporal Consistency (Eq. 16)

Stage 2 – Cross-Domain Registration (requires --registration_metrics_json):
    Registration Error (minimum MSE across frames), Anchor Frame Index

Stage 3 – Motion Faithfulness (always computed):
    EPE  (Eq. 13)  – endpoint error between MRI and anime optical flows
    DirSim (Eq. 14) – directional cosine similarity between flows
    Smooth (Eq. 15) – temporal smoothness of the anime flow field

NOTE: Metric implementations in this file match those in utils/metric_utils.py.
For new evaluation scripts, import from utils/metric_utils.py instead.
"""

import argparse
import importlib.util
import json
import os
import sys

import cv2
import numpy as np
import torch

# Add RAFT to path so the model can be loaded
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submodules", "RAFT"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submodules", "RAFT", "core"))
from raft import RAFT  # type: ignore

# Load metric_utils explicitly to avoid collision with RAFT's utils package.
_metric_utils_spec = importlib.util.spec_from_file_location(
    "metric_utils", os.path.join(os.path.dirname(__file__), "utils", "metric_utils.py")
)
if _metric_utils_spec is None or _metric_utils_spec.loader is None:
    raise RuntimeError("Failed to load utils/metric_utils.py")
_metric_utils = importlib.util.module_from_spec(_metric_utils_spec)
_metric_utils_spec.loader.exec_module(_metric_utils)
compute_stage1_metrics = _metric_utils.compute_stage1_metrics
compute_stage3_metrics = _metric_utils.compute_stage3_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Video I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_video_gray(video_path, resize=None):
    """Return a list of (H, W) uint8 grayscale frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if resize is not None:
            gray = cv2.resize(gray, resize, interpolation=cv2.INTER_LINEAR)
        frames.append(gray)
    cap.release()
    return frames


def load_video_bgr(video_path, resize=None):
    """Return a list of (H, W, 3) uint8 BGR frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize is not None:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    return frames


def temporal_downsample(frames, src_fps, tgt_fps):
    """
    Uniformly sub-sample *frames* from *src_fps* to approximately *tgt_fps*.
    Returns a new list with roughly len(frames) * (tgt_fps / src_fps) elements.
    """
    if src_fps <= 0 or tgt_fps <= 0 or src_fps == tgt_fps:
        return frames
    ratio = src_fps / tgt_fps
    indices = [int(round(i * ratio)) for i in range(int(len(frames) / ratio))]
    indices = [i for i in indices if i < len(frames)]
    return [frames[i] for i in indices]


def get_video_fps(video_path):
    """Return the frame-rate of *video_path* as a float."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def _load_raft(raft_path, device):
    model = RAFT(
        argparse.Namespace(
            model=str(raft_path),
            small=True,
            mixed_precision=False,
            alternate_corr=False,
        )
    )
    checkpoint = torch.load(str(raft_path), map_location=device)
    if "module." in list(checkpoint.keys())[0]:
        checkpoint = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in checkpoint.items()
        }
    model.load_state_dict(checkpoint)
    return model.to(device).eval()


def _compute_flow_raft(frame1_bgr, frame2_bgr, raft_model, device, flow_hw=(256, 256)):
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


def compute_consecutive_flows(frames_bgr, raft_model, device, flow_hw=(256, 256)):
    flows = []
    for i in range(len(frames_bgr) - 1):
        flow = _compute_flow_raft(
            frames_bgr[i], frames_bgr[i + 1], raft_model, device, flow_hw=flow_hw
        )
        flows.append(flow)
    return flows

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate A2M2A pipeline metrics as defined in the IEEE Access 2026 paper."
        )
    )

    # ── Inputs ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--pred_mri_video",
        type=str,
        required=True,
        help="Predicted MRI video produced by inference.py / run_pipeline_video.py.",
    )
    parser.add_argument(
        "--pred_anime_video",
        type=str,
        required=True,
        help="Predicted anime video produced by main.py (MRI-to-Anime stage).",
    )

    # ── Stage 1 (optional ground truth) ──────────────────────────────────────
    parser.add_argument(
        "--gt_mri_video",
        type=str,
        default=None,
        help=(
            "Ground-truth MRI video for Stage 1 metrics (MSE / PSNR / SSIM / Ltemp). "
            "Typically the original MRI recording from the USC-TIMIT dataset. "
            "The GT video is temporally downsampled to match the predicted FPS."
        ),
    )

    # ── Stage 2 (from registration JSON saved by main.py) ────────────────────
    parser.add_argument(
        "--registration_metrics_json",
        type=str,
        default=None,
        help=(
            "Path to the JSON file with registration metrics saved by "
            "run_pipeline_video.py --save_metrics_path."
        ),
    )

    # ── RAFT ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--raft_model",
        type=str,
        default="submodules/RAFT/models/raft-small.pth",
        help="Path to RAFT model checkpoint.",
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Disable GPU (runs optical flow on CPU; much slower).",
    )
    parser.add_argument(
        "--flow_size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("H", "W"),
        help="Internal frame size for RAFT inference (default: 256 256).",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="If set, save all metric results to this JSON file.",
    )

    args = parser.parse_args()

    device = "cpu" if args.no_gpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1 – MRI Reconstruction
    # ─────────────────────────────────────────────────────────────────────────
    if args.gt_mri_video:
        print("\n[Stage 1] Computing MRI reconstruction metrics...")
        gt_fps = get_video_fps(args.gt_mri_video)
        pred_fps = get_video_fps(args.pred_mri_video)

        gt_frames = load_video_gray(args.gt_mri_video)
        pred_frames = load_video_gray(args.pred_mri_video)

        # Temporally downsample GT to match predicted FPS
        if gt_fps > 0 and pred_fps > 0 and abs(gt_fps - pred_fps) > 0.5:
            print(
                f"  Downsampling GT from {gt_fps:.2f} fps to {pred_fps:.2f} fps "
                f"({len(gt_frames)} → ~{int(len(gt_frames)*pred_fps/gt_fps)} frames)"
            )
            gt_frames = temporal_downsample(gt_frames, gt_fps, pred_fps)

        # Align spatial size of pred to GT
        gt_h, gt_w = gt_frames[0].shape
        pred_frames = load_video_gray(args.pred_mri_video, resize=(gt_w, gt_h))

        stage1 = compute_stage1_metrics(gt_frames, pred_frames)
        results["stage1"] = stage1
        print(f"  MSE:                  {stage1['mse']:.4f}")
        print(f"  PSNR:                 {stage1['psnr']:.2f} dB")
        if stage1["ssim"] is not None:
            print(f"  SSIM:                 {stage1['ssim']:.4f}")
        print(f"  Temporal Consistency: {stage1['temporal_consistency']:.4f}")
    else:
        print("\n[Stage 1] Skipped — provide --gt_mri_video for MSE/PSNR/SSIM/Ltemp.")

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2 – Cross-Domain Registration
    # ─────────────────────────────────────────────────────────────────────────
    if args.registration_metrics_json and os.path.exists(args.registration_metrics_json):
        print("\n[Stage 2] Loading registration metrics...")
        with open(args.registration_metrics_json) as f:
            reg_metrics = json.load(f)
        results["stage2"] = reg_metrics
        print(f"  Registration Error (min MSE): {reg_metrics.get('registration_error', 'N/A')}")
        print(f"  Anchor Frame Index:           {reg_metrics.get('anchor_index', 'N/A')}")
    else:
        print(
            "\n[Stage 2] Skipped — provide --registration_metrics_json for "
            "registration error / anchor index."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3 – Motion Faithfulness
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Stage 3] Computing motion faithfulness metrics (EPE / DirSim / Smooth)...")
    if not os.path.exists(args.raft_model):
        print(
            f"  [Warning] RAFT model not found at '{args.raft_model}'. "
            "Stage 3 metrics skipped."
        )
    else:
        raft_model = _load_raft(args.raft_model, device)
        flow_hw = tuple(args.flow_size)  # (H, W)

        mri_frames_bgr = load_video_bgr(args.pred_mri_video)
        anime_frames_bgr = load_video_bgr(args.pred_anime_video)

        if len(mri_frames_bgr) < 2 or len(anime_frames_bgr) < 2:
            print("  Not enough frames for Stage 3 metrics (need ≥ 2).")
        else:
            print(
                f"  Computing flows: MRI ({len(mri_frames_bgr)} frames), "
                f"Anime ({len(anime_frames_bgr)} frames)..."
            )
            mri_flows = compute_consecutive_flows(
                mri_frames_bgr, raft_model, device, flow_hw=flow_hw
            )
            anime_flows = compute_consecutive_flows(
                anime_frames_bgr, raft_model, device, flow_hw=flow_hw
            )

            stage3 = compute_stage3_metrics(mri_flows, anime_flows)
            results["stage3"] = stage3
            print(f"  EPE:    {stage3['epe']:.4f} px")
            print(f"  DirSim: {stage3['dirsim']:.4f}  (higher is better)")
            print(f"  Smooth: {stage3['smooth']:.4f}  (lower is better)")

    # ─────────────────────────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────────────────────────
    if args.output_json:
        out_dir = os.path.dirname(os.path.abspath(args.output_json))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nMetrics saved to {args.output_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
