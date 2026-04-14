#!/usr/bin/env python3
"""
Metric Utilities for A2M2A Evaluation

This module provides centralized implementations of all evaluation metrics
described in the paper "Audio-Driven Articulatory 2D Animation Synthesis via 
MRI Generation and Cross-Domain Motion Transfer" (IEEE Access 2026).

All evaluation scripts (eval.py, eval_table1_baselines.py, run_batch_eval.sh, etc.)
should import from this module to ensure consistent metric computation.

Metrics are organized into three stages:
    Stage 1 – MRI Reconstruction Quality (Eq. 16)
    Stage 2 – Cross-Domain Registration Quality  
    Stage 3 – Motion Faithfulness (Eq. 13, 14, 15)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – MRI Reconstruction Quality Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_mse(gt_frames: List[np.ndarray], pred_frames: List[np.ndarray]) -> float:
    """
    Compute Mean Squared Error between ground truth and predicted frames.
    
    Args:
        gt_frames: List of (H, W) uint8 grayscale ground truth frames
        pred_frames: List of (H, W) uint8 grayscale predicted frames
    
    Returns:
        MSE value (lower is better)
    """
    T = min(len(gt_frames), len(pred_frames))
    if T == 0:
        raise ValueError("No frames to compare.")
    
    gt = [g.astype(np.float64) for g in gt_frames[:T]]
    pred = [p.astype(np.float64) for p in pred_frames[:T]]
    
    mse_vals = [np.mean((g - p) ** 2) for g, p in zip(gt, pred)]
    return float(np.mean(mse_vals))


def compute_psnr(mse: float, max_value: float = 255.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio from MSE.
    
    Args:
        mse: Mean squared error
        max_value: Maximum pixel value (default: 255 for uint8)
    
    Returns:
        PSNR in dB (higher is better)
    """
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(max_value ** 2 / mse)


def compute_ssim(gt_frames: List[np.ndarray], pred_frames: List[np.ndarray]) -> Optional[float]:
    """
    Compute Structural Similarity Index between ground truth and predicted frames.
    
    Args:
        gt_frames: List of (H, W) uint8 grayscale ground truth frames
        pred_frames: List of (H, W) uint8 grayscale predicted frames
    
    Returns:
        SSIM value in [0, 1] (higher is better), or None if scikit-image not available
    """
    T = min(len(gt_frames), len(pred_frames))
    if T == 0:
        raise ValueError("No frames to compare.")
    
    try:
        from skimage.metrics import structural_similarity
        
        gt = [g.astype(np.float64) for g in gt_frames[:T]]
        pred = [p.astype(np.float64) for p in pred_frames[:T]]
        
        ssim_vals = [
            structural_similarity(g, p, data_range=255.0)
            for g, p in zip(gt, pred)
        ]
        return float(np.mean(ssim_vals))
    except ImportError:
        print("  [Warning] scikit-image not available; SSIM not computed.")
        return None


def compute_temporal_consistency(
    gt_frames: List[np.ndarray], 
    pred_frames: List[np.ndarray]
) -> float:
    """
    Compute Temporal Consistency metric (Eq. 16 in paper).
    
    Measures how well temporal changes in predicted frames match ground truth.
    
    Formula:
        L_temp = (1/(T-1)) * Σ_t || (pred_{t+1} - pred_t) - (gt_{t+1} - gt_t) ||_2
    
    Args:
        gt_frames: List of (H, W) uint8 grayscale ground truth frames
        pred_frames: List of (H, W) uint8 grayscale predicted frames
    
    Returns:
        Temporal consistency error (lower is better)
    """
    T = min(len(gt_frames), len(pred_frames))
    if T < 2:
        return 0.0
    
    gt = [g.astype(np.float64) for g in gt_frames[:T]]
    pred = [p.astype(np.float64) for p in pred_frames[:T]]
    
    ltemp_vals = []
    for t in range(T - 1):
        pred_diff = pred[t + 1] - pred[t]
        gt_diff = gt[t + 1] - gt[t]
        ltemp_vals.append(float(np.linalg.norm(pred_diff - gt_diff)))
    
    return float(np.mean(ltemp_vals)) if ltemp_vals else 0.0


def compute_stage1_metrics(
    gt_frames: List[np.ndarray], 
    pred_frames: List[np.ndarray]
) -> dict:
    """
    Compute all Stage 1 metrics (MRI reconstruction quality).
    
    Args:
        gt_frames: List of (H, W) uint8 grayscale ground truth frames
        pred_frames: List of (H, W) uint8 grayscale predicted frames
    
    Returns:
        Dictionary with keys: mse, psnr, ssim, temporal_consistency
    """
    mse = compute_mse(gt_frames, pred_frames)
    psnr = compute_psnr(mse)
    ssim = compute_ssim(gt_frames, pred_frames)
    temporal_consistency = compute_temporal_consistency(gt_frames, pred_frames)
    
    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "temporal_consistency": temporal_consistency,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – Motion Faithfulness Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_epe(flow_mri: np.ndarray, flow_anime: np.ndarray) -> float:
    """
    Compute Endpoint Error (EPE) between two optical flow fields (Eq. 13).
    
    Measures average L2 distance between corresponding flow vectors.
    
    Formula:
        EPE = (1/(H*W)) * Σ_i || F_MRI(i) - F_anime(i) ||_2
    
    Args:
        flow_mri: Optical flow array (H, W, 2) from MRI video
        flow_anime: Optical flow array (H, W, 2) from anime video
    
    Returns:
        EPE value in pixels (lower is better)
    """
    if flow_mri.shape != flow_anime.shape:
        h, w = flow_mri.shape[:2]
        flow_anime = cv2.resize(flow_anime, (w, h))
    
    # Compute element-wise L2 norm
    diff = flow_mri - flow_anime
    epe = float(np.mean(np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)))
    
    return epe


def compute_dirsim(flow_mri: np.ndarray, flow_anime: np.ndarray) -> float:
    """
    Compute Directional Similarity using cosine similarity (Eq. 14).
    
    Measures how well flow directions align between MRI and anime.
    
    Formula:
        DirSim = (1/(H*W)) * Σ_i (F_MRI(i) · F_anime(i)) / (||F_MRI(i)|| * ||F_anime(i)||)
    
    Args:
        flow_mri: Optical flow array (H, W, 2) from MRI video
        flow_anime: Optical flow array (H, W, 2) from anime video
    
    Returns:
        DirSim value in [-1, 1] (higher is better)
            +1: Perfect alignment
             0: Perpendicular
            -1: Opposite directions
    """
    if flow_mri.shape != flow_anime.shape:
        h, w = flow_mri.shape[:2]
        flow_anime = cv2.resize(flow_anime, (w, h))
    
    # Compute cosine similarity with epsilon to avoid division by zero
    eps = 1e-8
    dot = np.sum(flow_mri * flow_anime, axis=-1)
    norm_mri = np.sqrt(np.sum(flow_mri ** 2, axis=-1)) + eps
    norm_anime = np.sqrt(np.sum(flow_anime ** 2, axis=-1)) + eps
    
    cosine_sim = dot / (norm_mri * norm_anime)
    dirsim = float(np.mean(cosine_sim))
    
    return dirsim


def compute_smoothness(flow_sequence: List[np.ndarray]) -> float:
    """
    Compute temporal smoothness of optical flow sequence (Eq. 15).
    
    Measures temporal consistency of flow fields across frames.
    
    Formula:
        Smooth = (1/(T-1)) * Σ_t mean(|| flow_{t+1} - flow_t ||_2)
    
    Args:
        flow_sequence: List of (H, W, 2) optical flow arrays
    
    Returns:
        Smoothness error in pixels (lower is better)
    """
    if len(flow_sequence) < 2:
        return 0.0
    
    smooth_vals = []
    for t in range(len(flow_sequence) - 1):
        flow_curr = flow_sequence[t]
        flow_next = flow_sequence[t + 1]
        
        # Ensure same shape
        if flow_curr.shape != flow_next.shape:
            h, w = flow_curr.shape[:2]
            flow_next = cv2.resize(flow_next, (w, h))
        
        # Compute per-pixel L2 norm of temporal difference
        diff = flow_next - flow_curr
        per_pixel_norm = np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)
        smooth_vals.append(float(np.mean(per_pixel_norm)))
    
    return float(np.mean(smooth_vals)) if smooth_vals else 0.0


def compute_motion_activity_ratio(
    flow_mri: np.ndarray,
    flow_anime: np.ndarray,
    tau: float = 0.5,
) -> float:
    """
    Compute Motion Activity Ratio (R_act).

    Formula:
        R_act = sum_i 1(||F_anim(i)|| > tau) / sum_i 1(||F_mri(i)|| > tau)

    Args:
        flow_mri: Optical flow array (H, W, 2) from MRI video
        flow_anime: Optical flow array (H, W, 2) from anime video
        tau: Motion magnitude threshold in pixels

    Returns:
        Motion activity ratio. Returns 0.0 if MRI has no active-motion pixels.
    """
    if flow_mri.shape != flow_anime.shape:
        h, w = flow_mri.shape[:2]
        flow_anime = cv2.resize(flow_anime, (w, h))

    mag_mri = np.sqrt(flow_mri[..., 0] ** 2 + flow_mri[..., 1] ** 2)
    mag_anime = np.sqrt(flow_anime[..., 0] ** 2 + flow_anime[..., 1] ** 2)

    active_mri = int(np.sum(mag_mri > tau))
    active_anime = int(np.sum(mag_anime > tau))
    if active_mri == 0:
        return 0.0
    return float(active_anime / active_mri)


def compute_motion_coverage_ratio(
    flow_mri: np.ndarray,
    flow_anime: np.ndarray,
    tau: float = 0.5,
) -> float:
    """
    Compute Motion Coverage Ratio (R_cov).

    Formula:
        R_cov = sum_i 1(||F_mri(i)|| > tau AND ||F_anim(i)|| > tau) /
                sum_i 1(||F_mri(i)|| > tau)

    Args:
        flow_mri: Optical flow array (H, W, 2) from MRI video
        flow_anime: Optical flow array (H, W, 2) from anime video
        tau: Motion magnitude threshold in pixels

    Returns:
        Motion coverage ratio. Returns 0.0 if MRI has no active-motion pixels.
    """
    if flow_mri.shape != flow_anime.shape:
        h, w = flow_mri.shape[:2]
        flow_anime = cv2.resize(flow_anime, (w, h))

    mag_mri = np.sqrt(flow_mri[..., 0] ** 2 + flow_mri[..., 1] ** 2)
    mag_anime = np.sqrt(flow_anime[..., 0] ** 2 + flow_anime[..., 1] ** 2)

    active_mri = mag_mri > tau
    active_anime = mag_anime > tau
    denom = int(np.sum(active_mri))
    if denom == 0:
        return 0.0
    overlap = int(np.sum(active_mri & active_anime))
    return float(overlap / denom)


def compute_stage3_metrics(
    mri_flows: List[np.ndarray], 
    anime_flows: List[np.ndarray],
    tau: float = 0.5,
) -> dict:
    """
    Compute all Stage 3 metrics (motion faithfulness).
    
    Args:
        mri_flows: List of (H, W, 2) consecutive-frame optical flows from MRI video
        anime_flows: List of (H, W, 2) consecutive-frame optical flows from anime video
    
    Returns:
        Dictionary with keys: epe, dirsim, smooth, motion_activity_ratio, motion_coverage_ratio
    """
    T = min(len(mri_flows), len(anime_flows))
    if T == 0:
        raise ValueError("No flow pairs to compare.")
    
    mri_f = mri_flows[:T]
    anime_f = anime_flows[:T]
    
    # Align spatial dimensions if needed
    Hm, Wm = mri_f[0].shape[:2]
    Ha, Wa = anime_f[0].shape[:2]
    if (Hm, Wm) != (Ha, Wa):
        anime_f = [cv2.resize(f, (Wm, Hm)) for f in anime_f]
        # Scale flow magnitudes
        for f in anime_f:
            f[..., 0] *= Wm / Wa
            f[..., 1] *= Hm / Ha
    
    # Compute EPE and DirSim for each frame pair
    epe_vals = []
    dirsim_vals = []
    motion_ratio_vals = []
    coverage_ratio_vals = []
    for f_mri, f_anime in zip(mri_f, anime_f):
        epe_vals.append(compute_epe(f_mri, f_anime))
        dirsim_vals.append(compute_dirsim(f_mri, f_anime))
        motion_ratio_vals.append(compute_motion_activity_ratio(f_mri, f_anime, tau=tau))
        coverage_ratio_vals.append(compute_motion_coverage_ratio(f_mri, f_anime, tau=tau))
    
    # Compute smoothness
    smooth = compute_smoothness(anime_f)
    
    return {
        "epe": float(np.mean(epe_vals)),
        "dirsim": float(np.mean(dirsim_vals)),
        "smooth": smooth,
        "motion_activity_ratio": float(np.mean(motion_ratio_vals)),
        "motion_coverage_ratio": float(np.mean(coverage_ratio_vals)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def resize_flows(flows: List[np.ndarray], target_wh: Tuple[int, int]) -> List[np.ndarray]:
    """
    Resize a list of optical flow arrays to target dimensions.
    
    Args:
        flows: List of (H, W, 2) flow arrays
        target_wh: Target dimensions (width, height)
    
    Returns:
        List of resized flow arrays with scaled magnitudes
    """
    W, H = target_wh
    resized = []
    
    for f in flows:
        fh, fw = f.shape[:2]
        r = cv2.resize(f, (W, H))
        # Scale flow magnitudes
        r[..., 0] *= W / fw
        r[..., 1] *= H / fh
        resized.append(r)
    
    return resized


def warp_frame_with_flow(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Warp an image using an optical flow field.
    
    Args:
        image: Input image (H, W, C) or (H, W)
        flow: Optical flow (H, W, 2) where flow[:,:,0] is x-displacement, 
              flow[:,:,1] is y-displacement
    
    Returns:
        Warped image with same shape as input
    """
    h, w = flow.shape[:2]
    
    # Resize image if needed
    if image.shape[:2] != (h, w):
        image = cv2.resize(image, (w, h))
    
    # Create coordinate maps
    flow_map = np.zeros((h, w, 2), dtype=np.float32)
    flow_map[:, :, 0] = np.arange(w)  # x coordinates
    flow_map[:, :, 1] = np.arange(h)[:, np.newaxis]  # y coordinates
    
    # Add flow to get target coordinates
    flow_map += flow
    
    # Warp using remap
    warped = cv2.remap(
        image, 
        flow_map, 
        None, 
        cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return warped


# ─────────────────────────────────────────────────────────────────────────────
# Metric Summary Functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    gt_mri_frames: Optional[List[np.ndarray]] = None,
    pred_mri_frames: Optional[List[np.ndarray]] = None,
    mri_flows: Optional[List[np.ndarray]] = None,
    anime_flows: Optional[List[np.ndarray]] = None,
) -> dict:
    """
    Compute all available metrics based on provided inputs.
    
    Args:
        gt_mri_frames: Ground truth MRI frames for Stage 1 metrics
        pred_mri_frames: Predicted MRI frames for Stage 1 metrics
        mri_flows: MRI optical flows for Stage 3 metrics
        anime_flows: Anime optical flows for Stage 3 metrics
    
    Returns:
        Dictionary containing all computed metrics
    """
    results = {}
    
    # Stage 1 metrics (if ground truth MRI available)
    if gt_mri_frames is not None and pred_mri_frames is not None:
        stage1 = compute_stage1_metrics(gt_mri_frames, pred_mri_frames)
        results.update(stage1)
    
    # Stage 3 metrics (if optical flows available)
    if mri_flows is not None and anime_flows is not None:
        stage3 = compute_stage3_metrics(mri_flows, anime_flows)
        results.update(stage3)
    
    return results


def print_metrics(metrics: dict, prefix: str = "") -> None:
    """
    Pretty-print metrics dictionary.
    
    Args:
        metrics: Dictionary of metric names and values
        prefix: Optional prefix for each line
    """
    print(f"{prefix}Metrics:")
    
    # Stage 1 metrics
    if "mse" in metrics:
        print(f"{prefix}  MSE:  {metrics['mse']:.6f}")
    if "psnr" in metrics:
        psnr_str = f"{metrics['psnr']:.2f} dB" if metrics['psnr'] != float("inf") else "inf"
        print(f"{prefix}  PSNR: {psnr_str}")
    if "ssim" in metrics and metrics["ssim"] is not None:
        print(f"{prefix}  SSIM: {metrics['ssim']:.4f}")
    if "temporal_consistency" in metrics:
        print(f"{prefix}  Temporal Consistency: {metrics['temporal_consistency']:.6f}")
    
    # Stage 3 metrics
    if "epe" in metrics:
        print(f"{prefix}  EPE:    {metrics['epe']:.6f}")
    if "dirsim" in metrics:
        print(f"{prefix}  DirSim: {metrics['dirsim']:.6f}")
    if "smooth" in metrics:
        print(f"{prefix}  Smooth: {metrics['smooth']:.6f}")
    if "motion_activity_ratio" in metrics:
        print(f"{prefix}  MotionActivityRatio: {metrics['motion_activity_ratio']:.6f}")
    if "motion_coverage_ratio" in metrics:
        print(f"{prefix}  MotionCoverageRatio: {metrics['motion_coverage_ratio']:.6f}")
