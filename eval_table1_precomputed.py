#!/usr/bin/env python3
"""
TABLE1 eval-only script for precomputed visualizations.

Reads existing outputs under:
  eval_output_table1/subXXX/<clip_id>/visualizations/<method_dir>/

Expected method content:
  temporal_sequence/anime_warped_*.png

The script computes Stage-3 metrics only (EPE, DirSim, Smooth, MotionActivityRatio,
MotionCoverageRatio) by:
1) loading GT MRI frames from dataset video,
2) loading precomputed anime frame sequence from temporal_sequence,
3) computing optical flows with method-specific flow backend inferred from method name.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))
sys.path.insert(0, str(_script_dir / "submodules" / "RAFT"))
sys.path.insert(0, str(_script_dir / "submodules" / "RAFT" / "core"))

from raft import RAFT  # type: ignore

# Load metric_utils explicitly (avoid collision with RAFT's utils package).
_metric_utils_spec = importlib.util.spec_from_file_location(
    "metric_utils", str(_script_dir / "utils" / "metric_utils.py")
)
if _metric_utils_spec is None or _metric_utils_spec.loader is None:
    raise RuntimeError("Failed to load utils/metric_utils.py")
_metric_utils = importlib.util.module_from_spec(_metric_utils_spec)
_metric_utils_spec.loader.exec_module(_metric_utils)
compute_stage3_metrics = _metric_utils.compute_stage3_metrics


def _load_video_frames(video_path: Path) -> List[np.ndarray]:
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


def _load_anime_frames_from_dir(temporal_dir: Path) -> List[np.ndarray]:
    pngs = sorted(temporal_dir.glob("anime_warped_*.png"))
    frames: List[np.ndarray] = []
    for p in pngs:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    return frames


def _compute_flow_hornschunck(frame1_bgr: np.ndarray, frame2_bgr: np.ndarray) -> np.ndarray:
    frame1 = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    frame2 = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def _compute_flow_tvl1(frame1_bgr: np.ndarray, frame2_bgr: np.ndarray) -> np.ndarray:
    frame1 = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    frame2 = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 5, 15, 10, 5, 1.2, 0)


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
            (k[7:] if k.startswith("module.") else k): v
            for k, v in checkpoint.items()
        }
    model.load_state_dict(checkpoint)
    return model.to(device).eval()


def _compute_flow_raft(
    frame1_bgr: np.ndarray,
    frame2_bgr: np.ndarray,
    raft_model: torch.nn.Module,
    device: str,
    flow_hw: Tuple[int, int] = (256, 256),
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


def _infer_flow_mode_from_method(method_dir_name: str) -> str:
    if "Deep_Flow" in method_dir_name:
        return "deep"
    if "TV-L1" in method_dir_name:
        return "tvl1"
    if "Horn-Schunck" in method_dir_name:
        return "horn"
    raise ValueError(f"Unknown flow method in directory name: {method_dir_name}")


def _compute_consecutive_flows(
    frames: List[np.ndarray],
    flow_mode: str,
    raft_model: Optional[torch.nn.Module],
    device: str,
) -> List[np.ndarray]:
    flows: List[np.ndarray] = []
    for i in range(len(frames) - 1):
        if flow_mode == "horn":
            flow = _compute_flow_hornschunck(frames[i], frames[i + 1])
        elif flow_mode == "tvl1":
            flow = _compute_flow_tvl1(frames[i], frames[i + 1])
        elif flow_mode == "deep":
            if raft_model is None:
                raise RuntimeError("RAFT model is required for Deep Flow methods")
            flow = _compute_flow_raft(frames[i], frames[i + 1], raft_model, device)
        else:
            raise ValueError(f"Unsupported flow mode: {flow_mode}")
        flows.append(flow)
    return flows


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


def run_evaluation(
    eval_root: Path,
    dataset_root: Path,
    subjects: List[str],
    clip_ids: List[str],
    method_dirs: Optional[List[str]],
    output_json_name: str,
    motion_threshold: float,
    num_frames: Optional[int],
    device: str,
    resume: bool,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, falling back to CPU.")
        device = "cpu"

    need_raft = False
    if method_dirs:
        need_raft = any("Deep_Flow" in m for m in method_dirs)

    raft_model = None
    if need_raft:
        raft_path = _script_dir / "submodules" / "RAFT" / "models" / "raft-small.pth"
        if not raft_path.is_file():
            raise FileNotFoundError(f"RAFT checkpoint not found: {raft_path}")
        print(f"Loading RAFT model: {raft_path}")
        raft_model = _load_raft(raft_path, device)

    total = 0
    done = 0
    skip = 0
    fail = 0
    method_values: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for sub in subjects:
        sub_id = sub if sub.startswith("sub") else f"sub{sub}"
        for clip_id in clip_ids:
            clip_dir = eval_root / sub_id / clip_id
            vis_root = clip_dir / "visualizations"
            out_json = clip_dir / output_json_name
            total += 1

            if not vis_root.is_dir():
                print(f"[SKIP] {sub_id}/{clip_id} - visualizations missing")
                skip += 1
                continue

            if resume and out_json.is_file():
                print(f"[DONE] {sub_id}/{clip_id} - {output_json_name} exists")
                skip += 1
                continue

            gt_video = _resolve_gt_video(dataset_root, sub_id, clip_id)
            if gt_video is None:
                print(f"[FAIL] {sub_id}/{clip_id} - GT MRI video not found")
                fail += 1
                continue

            try:
                mri_frames = _load_video_frames(gt_video)
                if num_frames is not None:
                    mri_frames = mri_frames[:num_frames]
                if len(mri_frames) < 2:
                    raise RuntimeError(f"Not enough MRI frames: {len(mri_frames)}")

                target_methods = method_dirs if method_dirs else sorted(
                    [d.name for d in vis_root.iterdir() if d.is_dir()]
                )
                clip_results = {}
                method_errors = {}
                local_raft_loaded = raft_model

                for method_dir_name in target_methods:
                    method_dir = vis_root / method_dir_name
                    if not method_dir.is_dir():
                        method_errors[method_dir_name] = "missing method directory"
                        continue

                    try:
                        flow_mode = _infer_flow_mode_from_method(method_dir_name)
                        if flow_mode == "deep" and local_raft_loaded is None:
                            raft_path = _script_dir / "submodules" / "RAFT" / "models" / "raft-small.pth"
                            if not raft_path.is_file():
                                raise FileNotFoundError(f"RAFT checkpoint not found: {raft_path}")
                            local_raft_loaded = _load_raft(raft_path, device)

                        anime_frames = _load_anime_frames_from_dir(method_dir / "temporal_sequence")
                        if num_frames is not None:
                            anime_frames = anime_frames[:num_frames]

                        if len(anime_frames) < 2:
                            raise RuntimeError(
                                "missing/insufficient temporal_sequence/anime_warped_*.png"
                            )

                        t = min(len(mri_frames), len(anime_frames))
                        mri_eval = mri_frames[:t]
                        anime_eval = anime_frames[:t]

                        mri_flows = _compute_consecutive_flows(
                            mri_eval, flow_mode, local_raft_loaded, device
                        )
                        anime_flows = _compute_consecutive_flows(
                            anime_eval, flow_mode, local_raft_loaded, device
                        )
                        metrics = compute_stage3_metrics(
                            mri_flows, anime_flows, tau=motion_threshold
                        )
                        clip_results[method_dir_name] = {
                            "EPE": metrics["epe"],
                            "DirSim": metrics["dirsim"],
                            "Smooth": metrics["smooth"],
                            "MotionActivityRatio": metrics["motion_activity_ratio"],
                            "MotionCoverageRatio": metrics["motion_coverage_ratio"],
                            "num_mri_frames": len(mri_eval),
                            "num_anime_frames": len(anime_eval),
                        }
                        method_values[method_dir_name]["epe"].append(metrics["epe"])
                        method_values[method_dir_name]["dirsim"].append(metrics["dirsim"])
                        method_values[method_dir_name]["smooth"].append(metrics["smooth"])
                        method_values[method_dir_name]["actratio"].append(metrics["motion_activity_ratio"])
                        method_values[method_dir_name]["covratio"].append(metrics["motion_coverage_ratio"])
                    except Exception as e:
                        method_errors[method_dir_name] = str(e)

                payload = {
                    "clip": f"{sub_id}/{clip_id}",
                    "motion_threshold": motion_threshold,
                    "results": clip_results,
                }
                if method_errors:
                    payload["errors"] = method_errors

                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_json.write_text(json.dumps(payload, indent=2))
                done += 1
                print(
                    f"[OK] {sub_id}/{clip_id} - methods: {len(clip_results)}"
                    + (f", errors: {len(method_errors)}" if method_errors else "")
                )
            except Exception as e:
                fail += 1
                print(f"[FAIL] {sub_id}/{clip_id} - {e}")

    print("\n" + "=" * 85)
    print("TABLE1 PRECOMPUTED EVALUATION SUMMARY")
    print("=" * 85)
    print(
        f"{'Method':<28} {'Count':>6} {'EPE':>10} {'DirSim':>10} "
        f"{'Smooth':>10} {'ActRatio':>10} {'CovRatio':>10}"
    )
    print("-" * 85)
    for method in sorted(method_values.keys()):
        vals = method_values[method]
        c = len(vals["epe"])
        epe = sum(vals["epe"]) / c if c else float("nan")
        dirsim = sum(vals["dirsim"]) / c if c else float("nan")
        smooth = sum(vals["smooth"]) / c if c else float("nan")
        act = sum(vals["actratio"]) / c if c else float("nan")
        cov = sum(vals["covratio"]) / c if c else float("nan")
        print(
            f"{method:<28} {c:>6} {epe:>10.6f} {dirsim:>10.6f} "
            f"{smooth:>10.6f} {act:>10.6f} {cov:>10.6f}"
        )
    print("-" * 85)
    print(f"Total clips: {total}, done: {done}, skipped: {skip}, failed: {fail}")
    print("=" * 85)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TABLE1 outputs from precomputed visualizations.")
    parser.add_argument("--eval_root", type=str, default="eval_output_table1")
    parser.add_argument("--dataset_root", type=str, default="/ssd1tb_00/dataset/mri_data")
    parser.add_argument("--subjects", type=str, default="002 009 014 025 028 038 039 057 067")
    parser.add_argument(
        "--clip_ids",
        type=str,
        default="01_vcv1_r1 02_vcv2_r1 03_vcv3_r1 04_bvt_r1 05_shibboleth_r1 "
                "06_rainbow_r1 07_grandfather1_r1 08_grandfather2_r1 09_northwind1_r1 10_northwind2_r1",
    )
    parser.add_argument(
        "--method_keys",
        type=str,
        default="",
        help="Comma-separated method dir names, e.g. Ours_Deep_Flow,LoFTR_Deep_Flow",
    )
    parser.add_argument("--output_json_name", type=str, default="eval_metrics.json")
    parser.add_argument("--motion_threshold", type=float, default=0.5)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--resume", action="store_true", help="Skip clip if output JSON exists")
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    discovered_subjects, discovered_clips = _discover_subjects_and_clips(eval_root)

    raw_methods = args.method_keys.strip()
    if raw_methods and raw_methods.lower() != "all":
        methods = [m.strip() for m in raw_methods.split(",") if m.strip()]
    else:
        methods = None

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

    run_evaluation(
        eval_root=eval_root,
        dataset_root=Path(args.dataset_root),
        subjects=subjects,
        clip_ids=clip_ids,
        method_dirs=methods,
        output_json_name=args.output_json_name,
        motion_threshold=args.motion_threshold,
        num_frames=args.num_frames,
        device=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
