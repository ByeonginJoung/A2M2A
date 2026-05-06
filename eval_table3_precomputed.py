#!/usr/bin/env python3
"""
TABLE3 eval-only script for precomputed visualizations.

Loads already-saved frame assets and computes Stage-3 metrics only.
No anime generation/deformation is executed in this script.

Expected layout:
  <eval_root>/subXXX/<clip_id>/visualizations/<ablation>/temporal_sequence/
    - anime_warped_*.png

For protocol parity with TABLE1 precomputed evaluation, MRI is always loaded
from dataset_root GT video (not from saved mri_*.png).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch

_script_dir = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(_script_dir / "submodules" / "RAFT"))
sys.path.insert(0, str(_script_dir / "submodules" / "RAFT" / "core"))
from raft import RAFT  # type: ignore

_metric_utils_spec = importlib.util.spec_from_file_location(
    "metric_utils", str(_script_dir / "utils" / "metric_utils.py")
)
if _metric_utils_spec is None or _metric_utils_spec.loader is None:
    raise RuntimeError("Failed to load utils/metric_utils.py")
_metric_utils = importlib.util.module_from_spec(_metric_utils_spec)
_metric_utils_spec.loader.exec_module(_metric_utils)
compute_stage3_metrics = _metric_utils.compute_stage3_metrics

PROTOCOL = "table3_precomputed_v1"
ABLATION_DEFAULTS = ["none", "preprocess_only", "preprocess_anchor", "full"]
ABLATION_FLAGS = {
    "none": (0, 0, 0),
    "preprocess_only": (1, 0, 0),
    "preprocess_anchor": (1, 1, 0),
    "full": (1, 1, 1),
}


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _safe_float(v, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _load_video_bgr(video_path: Path) -> List[np.ndarray]:
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


def _load_image_sequence(seq_dir: Path, prefix: str) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for p in sorted(seq_dir.glob(f"{prefix}*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    return frames


def _resolve_gt_video(dataset_root: Path, sub_id: str, clip_id: str) -> Optional[Path]:
    stem = f"{sub_id}_2drt_{clip_id}_video"
    base = dataset_root / sub_id / "2drt" / "video"
    for ext in (".mp4", ".avi"):
        p = base / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _discover_subjects_and_clips(eval_root: Path) -> Tuple[List[str], List[str]]:
    subjects: Set[str] = set()
    clips: Set[str] = set()
    if not eval_root.is_dir():
        return [], []
    for sub_dir in eval_root.iterdir():
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub"):
            continue
        subjects.add(sub_dir.name)
        for clip_dir in sub_dir.iterdir():
            if clip_dir.is_dir():
                clips.add(clip_dir.name)
    return sorted(subjects), sorted(clips)


def _load_raft(raft_path: Path, device: str) -> torch.nn.Module:
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
            (k[7:] if k.startswith("module.") else k): v for k, v in checkpoint.items()
        }
    model.load_state_dict(checkpoint)
    return model.to(device).eval()


def _compute_flow_raft(
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


def _compute_consecutive_flows(
    frames: Sequence[np.ndarray],
    raft_model: torch.nn.Module,
    device: str,
    flow_hw: Tuple[int, int],
) -> List[np.ndarray]:
    flows: List[np.ndarray] = []
    for i in range(len(frames) - 1):
        flows.append(_compute_flow_raft(frames[i], frames[i + 1], raft_model, device, flow_hw))
    return flows


def run_evaluation(
    eval_root: Path,
    dataset_root: Path,
    subjects: List[str],
    clip_ids: List[str],
    ablations: List[str],
    motion_threshold: float,
    device: str,
    flow_hw: Tuple[int, int],
    resume: bool,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, falling back to CPU.")
        device = "cpu"

    raft_path = _script_dir / "submodules" / "RAFT" / "models" / "raft-small.pth"
    if not raft_path.is_file():
        raise FileNotFoundError(f"RAFT checkpoint not found: {raft_path}")
    print(f"Loading RAFT model: {raft_path}")
    raft_model = _load_raft(raft_path, device)

    detail_json = eval_root / "table3_detail_precomputed.json"
    per_clip_csv = eval_root / "table3_per_clip_precomputed.csv"
    summary_csv = eval_root / "table3_summary_precomputed.csv"
    summary_txt = eval_root / "table3_summary_precomputed.txt"

    legacy_detail = eval_root / "table3_detail.json"
    legacy_cache = {}
    if legacy_detail.is_file():
        try:
            legacy_cache = json.loads(legacy_detail.read_text())
        except Exception:
            legacy_cache = {}

    cache: Dict[str, Dict] = {}
    if resume and detail_json.is_file():
        try:
            cache = json.loads(detail_json.read_text())
        except Exception:
            cache = {}

    rows: List[Dict] = []
    total = 0
    done = 0
    skip = 0
    fail = 0

    for sub in subjects:
        sub_id = sub if sub.startswith("sub") else f"sub{sub}"
        for clip_id in clip_ids:
            for ab_name in ablations:
                total += 1
                cache_key = f"{sub_id}/{clip_id}::{ab_name}"

                if resume and cache_key in cache:
                    c = cache[cache_key]
                    if (
                        c.get("protocol") == PROTOCOL
                        and c.get("flow_method") == "deep_flow"
                        and float(c.get("motion_threshold", motion_threshold)) == motion_threshold
                    ):
                        rows.append(c)
                        skip += 1
                        continue

                seq_dir = eval_root / sub_id / clip_id / "visualizations" / ab_name / "temporal_sequence"
                if not seq_dir.is_dir():
                    continue

                try:
                    anime_frames = _load_image_sequence(seq_dir, "anime_warped_")
                    if len(anime_frames) < 2:
                        raise RuntimeError("missing/insufficient anime_warped_*.png")

                    gt_video = _resolve_gt_video(dataset_root, sub_id, clip_id)
                    if gt_video is None:
                        raise RuntimeError("GT MRI video not found")
                    mri_frames = _load_video_bgr(gt_video)

                    n = min(len(mri_frames), len(anime_frames))
                    if n < 2:
                        raise RuntimeError("not enough aligned frames")

                    mri_eval = mri_frames[:n]
                    anime_eval = anime_frames[:n]
                    mri_flows = _compute_consecutive_flows(mri_eval, raft_model, device, flow_hw)
                    anime_flows = _compute_consecutive_flows(anime_eval, raft_model, device, flow_hw)
                    stage3 = compute_stage3_metrics(mri_flows, anime_flows, tau=motion_threshold)

                    pre, anch, bidi = ABLATION_FLAGS.get(ab_name, (None, None, None))
                    legacy = legacy_cache.get(cache_key, {}) if isinstance(legacy_cache, dict) else {}
                    row = {
                        "subject": sub_id,
                        "clip": clip_id,
                        "ablation": ab_name,
                        "preprocess": int(pre) if pre is not None else "",
                        "anchor": int(anch) if anch is not None else "",
                        "bidirectional": int(bidi) if bidi is not None else "",
                        "anchor_index": _safe_int(legacy.get("anchor_index", 0), 0),
                        "registration_error": _safe_float(legacy.get("registration_error", float("nan")), float("nan")),
                        "epe": float(stage3.get("epe")),
                        "dirsim_raw": float(stage3.get("dirsim")),
                        "dirsim": float(stage3.get("dirsim")),
                        "dirsim_scale": "raw_-1_1",
                        "smooth": float(stage3.get("smooth")),
                        "act_ratio": float(stage3.get("motion_activity_ratio")),
                        "cov_ratio": float(stage3.get("motion_coverage_ratio")),
                        "pre_scale_target": legacy.get("pre_scale_target", ""),
                        "warp_strategy": "precomputed_frames",
                        "mri_source": "gt_dataset_video",
                        "flow_method": "deep_flow",
                        "motion_threshold": motion_threshold,
                        "protocol": PROTOCOL,
                        "num_mri_frames": len(mri_eval),
                        "num_anime_frames": len(anime_eval),
                    }
                    rows.append(row)
                    cache[cache_key] = row
                    done += 1
                    print(f"[OK] {sub_id}/{clip_id}/{ab_name}")
                except Exception as e:
                    fail += 1
                    print(f"[FAIL] {sub_id}/{clip_id}/{ab_name} - {e}")

    detail_json.write_text(json.dumps(cache, indent=2))

    if not rows:
        print("No rows computed.")
        return

    fieldnames = [
        "subject","clip","ablation","preprocess","anchor","bidirectional",
        "anchor_index","registration_error","epe","dirsim","dirsim_raw","dirsim_scale","smooth",
        "act_ratio","cov_ratio","pre_scale_target","warp_strategy","mri_source",
        "flow_method","motion_threshold","protocol","num_mri_frames","num_anime_frames",
    ]
    with per_clip_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    by_ablation: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_ablation[r["ablation"]].append(r)

    summary_rows = []
    for ab in ablations:
        vals = by_ablation.get(ab, [])
        if not vals:
            continue
        pre, anch, bidi = ABLATION_FLAGS.get(ab, (None, None, None))

        def m(k: str) -> float:
            return float(np.mean([float(v[k]) for v in vals]))

        summary_rows.append({
            "ablation": ab,
            "preprocess": "✓" if pre == 1 else "",
            "anchor": "✓" if anch == 1 else "",
            "bidirectional": "✓" if bidi == 1 else "",
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
                "epe","dirsim","smooth","act_ratio","cov_ratio",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)

    header = "  {:<18} {:<10} {:<8} {:<13} {:>8} {:>10} {:>12} {:>10} {:>10} {:>10}".format(
        "Ablation", "Preprocess", "Anchor", "Bidirectional", "N", "EPE", "DirSim", "Smooth", "ActRatio", "CovRatio"
    )
    sep = "  " + "-" * (len(header) - 2)
    lines = [header, sep]
    for r in summary_rows:
        lines.append(
            "  {:<18} {:<10} {:<8} {:<13} {:>8} {:>10.4f} {:>12.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                r["ablation"], r["preprocess"], r["anchor"], r["bidirectional"],
                int(r["n_clips"]), r["epe"], r["dirsim"], r["smooth"], r["act_ratio"], r["cov_ratio"]
            )
        )
    lines.append(sep)

    with summary_txt.open("w") as f:
        f.write("TABLE3 PRECOMPUTED EVALUATION SUMMARY\n\n")
        for line in lines:
            f.write(line + "\n")

    for line in lines:
        print(line)
    print()
    print(f"Total combinations: {total}, done: {done}, skipped: {skip}, failed: {fail}")
    print(f"Per-clip CSV : {per_clip_csv}")
    print(f"Summary CSV  : {summary_csv}")
    print(f"Summary TXT  : {summary_txt}")
    print(f"Detail JSON  : {detail_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TABLE3 outputs from precomputed visualizations.")
    parser.add_argument("--eval_root", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default="/ssd1tb_00/dataset/mri_data")
    parser.add_argument("--subjects", type=str, default="all")
    parser.add_argument("--clip_ids", type=str, default="all")
    parser.add_argument(
        "--ablations",
        type=str,
        default="all",
        help="Space/comma-separated ablations (none, preprocess_only, preprocess_anchor, full) or 'all'.",
    )
    parser.add_argument("--motion_threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--flow_size", nargs=2, type=int, default=[256, 256], metavar=("H", "W"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    discovered_subjects, discovered_clips = _discover_subjects_and_clips(eval_root)

    raw_subjects = [s for s in args.subjects.split() if s]
    if len(raw_subjects) == 1 and raw_subjects[0].lower() == "all":
        subjects = discovered_subjects
    else:
        subjects = raw_subjects

    raw_clips = [c for c in args.clip_ids.split() if c]
    if len(raw_clips) == 1 and raw_clips[0].lower() == "all":
        clip_ids = discovered_clips
    else:
        clip_ids = raw_clips

    if not subjects:
        raise RuntimeError("No subjects to evaluate. Check --eval_root / --subjects.")
    if not clip_ids:
        raise RuntimeError("No clip_ids to evaluate. Check --eval_root / --clip_ids.")

    raw_ab = args.ablations.strip().replace(",", " ")
    if not raw_ab or raw_ab.lower() == "all":
        ablations = ABLATION_DEFAULTS
    else:
        ablations = [a for a in raw_ab.split() if a]

    flow_hw = (args.flow_size[0], args.flow_size[1])
    run_evaluation(
        eval_root=eval_root,
        dataset_root=Path(args.dataset_root),
        subjects=subjects,
        clip_ids=clip_ids,
        ablations=ablations,
        motion_threshold=args.motion_threshold,
        device=args.device,
        flow_hw=flow_hw,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
