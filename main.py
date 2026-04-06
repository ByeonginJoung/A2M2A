import cv2
import numpy as np
import argparse
import torch
import sys
import os

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

# Add RAFT submodule to Python path
# Must add RAFT/core to sys.path and import without submodules prefix (mimics RAFT's demo.py)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_raft_core = os.path.join(_script_dir, 'submodules', 'RAFT', 'core')
sys.path.insert(0, _raft_core)  # Insert at beginning so RAFT's utils takes priority

from raft import RAFT
from utils import flow_viz
from utils.flow_viz import flow_to_image

def read_video_frames(video_path):
    """Reads all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def preprocess_for_registration(ref_mri, target_mri_frame):
    """Common preprocessing steps for registration."""
    target_mri_frame = cv2.resize(
        target_mri_frame,
        (ref_mri.shape[1], ref_mri.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )
    ref_denoised = cv2.medianBlur(ref_mri, 5)
    target_denoised = cv2.medianBlur(target_mri_frame, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    ref_enhanced = clahe.apply(ref_denoised)
    target_enhanced = clahe.apply(target_denoised)
    return ref_enhanced, target_enhanced

def register_ecc(ref_img, target_img):
    """Register images using ECC."""
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 1000
    termination_eps = 1e-6
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    try:
        _, warp_matrix = cv2.findTransformECC(
            ref_img.astype(np.float32),
            target_img.astype(np.float32),
            warp_matrix, warp_mode, criteria
        )
        h, w = ref_img.shape
        target_warped = cv2.warpAffine(target_img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        mse = np.mean((ref_img.astype("float") - target_warped.astype("float")) ** 2)
        return warp_matrix, mse
    except cv2.error as exc:
        try:
            print("ECC failed:", exc)
            print("  ref_img: shape={}, dtype={}, min={}, max={}, mean={}, std={}".format(
                ref_img.shape, ref_img.dtype, float(np.min(ref_img)), float(np.max(ref_img)),
                float(np.mean(ref_img)), float(np.std(ref_img))
            ))
            print("  target_img: shape={}, dtype={}, min={}, max={}, mean={}, std={}".format(
                target_img.shape, target_img.dtype, float(np.min(target_img)), float(np.max(target_img)),
                float(np.mean(target_img)), float(np.std(target_img))
            ))
        except Exception:
            print("ECC failed and debug stats could not be computed.")
        return np.eye(2, 3, dtype=np.float32), float('inf')

def register_orb(ref_img, target_img):
    """Register images using ORB with RANSAC for robustness."""
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(ref_img, None)
    kp2, des2 = orb.detectAndCompute(target_img, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return np.eye(2, 3, dtype=np.float32), float('inf')

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    good_matches = matches[:50]

    if len(good_matches) < 4:
        return np.eye(2, 3, dtype=np.float32), float('inf')

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    matrix, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=3)
    
    if matrix is None:
        return np.eye(2, 3, dtype=np.float32), float('inf')

    h, w = ref_img.shape
    target_warped = cv2.warpAffine(target_img, matrix, (w, h))
    mse = np.mean((ref_img.astype("float") - target_warped.astype("float")) ** 2)
    
    return matrix, mse

def register_sitk(ref_img, target_img):
    """Register images using SimpleITK and mutual information."""
    if not SITK_AVAILABLE:
        print("SimpleITK not available, falling back to ECC.")
        return register_ecc(ref_img, target_img)

    ref_sitk = sitk.GetImageFromArray(ref_img.astype(np.float32))
    target_sitk = sitk.GetImageFromArray(target_img.astype(np.float32))

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    R.SetInterpolator(sitk.sitkLinear)
    
    initial_transform = sitk.CenteredTransformInitializer(ref_sitk, target_sitk, sitk.AffineTransform(2), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetInitialTransform(initial_transform)

    try:
        final_transform_sitk = R.Execute(ref_sitk, target_sitk)
        
        A = np.array(final_transform_sitk.GetMatrix()).reshape(2, 2)
        c = np.array(final_transform_sitk.GetCenter())
        t = np.array(final_transform_sitk.GetTranslation())
        
        offset = t - A.dot(c) + c
        warp_matrix = np.hstack([A, offset.reshape(2, 1)])
        
        h, w = ref_img.shape
        target_warped = cv2.warpAffine(target_img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        mse = np.mean((ref_img.astype("float") - target_warped.astype("float")) ** 2)
        
        return warp_matrix, mse
    except Exception:
        return np.eye(2, 3, dtype=np.float32), float('inf')

def get_registration_for_frame(ref_mri, target_mri_frame, mode='ecc'):
    """Top-level registration function that dispatches to the correct method."""
    ref_enhanced, target_enhanced = preprocess_for_registration(ref_mri, target_mri_frame)
    if mode == 'orb':
        return register_orb(ref_enhanced, target_enhanced)
    elif mode == 'sitk':
        return register_sitk(ref_enhanced, target_enhanced)
    else:
        return register_ecc(ref_enhanced, target_enhanced)

def get_optical_flow(frames, ref_frame_index, model_path="submodules/RAFT/models/raft-small.pth", device="cuda"):
    """
    Uses RAFT to estimate optical flow between a reference frame and all other frames.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=model_path, help="restore checkpoint")
    parser.add_argument('--small', action='store_true', default=True, help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    args = parser.parse_args(args=[])
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(device)
    model.eval()

    def preprocess_frame(frame, target_dim=(256, 256)):
        frame = cv2.resize(frame, target_dim)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
        return frame_tensor[None].to(device)

    flows = []
    ref_frame_tensor = preprocess_frame(frames[ref_frame_index])
    
    for i, frame in enumerate(frames):
        target_frame_tensor = preprocess_frame(frame)
        
        _, flow_up = model(ref_frame_tensor, target_frame_tensor, iters=20, test_mode=True)
        flow_numpy = flow_up.detach().permute(0, 2, 3, 1).cpu().numpy()[0]
        
        h_orig, w_orig = frames[0].shape[:2]
        h_small, w_small = flow_numpy.shape[:2]

        resized_flow = cv2.resize(flow_numpy, (w_orig, h_orig))

        resized_flow[:, :, 0] *= (w_orig / w_small)
        resized_flow[:, :, 1] *= (h_orig / h_small)
        flows.append(resized_flow)
        
        torch.cuda.empty_cache()
    
    return flows

def warp_anime_frame(ref_anime, rigid_transform, flow_field, target_dims):
    """
    Warps the reference anime image using the combined transformation.
    This involves a two-step mapping for each pixel in the output image:
    1. From the current frame's space back to the best-matched frame's space using the optical flow.
    2. From the best-matched frame's space to the canonical reference space using the rigid transform.
    """
    h, w, _ = ref_anime.shape
    x, y = np.meshgrid(np.arange(target_dims[1]), np.arange(target_dims[0]))

    # Step 1: Create a map from the current frame's coordinates back to the best-matched frame's coordinates.
    # The flow_field gives displacement from the best_frame to the current_frame.
    # To go backwards, we subtract the flow from the current coordinates.
    map_x_to_best_frame = x - flow_field[:, :, 0]
    map_y_to_best_frame = y - flow_field[:, :, 1]

    # Step 2: Map the coordinates from the best-matched frame's space to the canonical reference space.
    # The rigid_transform maps from the reference space to the best_frame space. We need the inverse.
    target_to_ref_transform = cv2.invertAffineTransform(rigid_transform)
    
    # Combine coordinates for efficient matrix multiplication
    coords_in_best_frame = np.stack([map_x_to_best_frame.flatten(), map_y_to_best_frame.flatten()])
    coords_in_best_frame_hom = np.vstack([coords_in_best_frame, np.ones(coords_in_best_frame.shape[1])])

    # Apply the rigid transformation to get coordinates in the canonical reference space
    coords_in_ref_anime = target_to_ref_transform @ coords_in_best_frame_hom

    # Reshape the coordinates back to the image dimensions
    final_map_x = coords_in_ref_anime[0].reshape(target_dims[0], target_dims[1])
    final_map_y = coords_in_ref_anime[1].reshape(target_dims[0], target_dims[1])

    # Step 3: Use the final mapping to sample pixels from the reference anime image.
    warped_anime = cv2.remap(ref_anime, final_map_x.astype(np.float32), final_map_y.astype(np.float32), cv2.INTER_LINEAR)

    return warped_anime.astype(np.uint8)

def visualize_flow_arrows(frame, flow, step=16):
    """Draws arrows on a frame to visualize optical flow."""
    vis_frame = frame.copy()
    h, w = frame.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    for (x1, y1), (x2, y2) in lines:
        if np.linalg.norm((x1-x2, y1-y2)) > 1: # Only draw arrows for significant motion
            cv2.arrowedLine(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
            
    return vis_frame

def main(ref_mri_path, ref_anime_path, target_video_path, output_video_path, registration_mode='ecc', debug=False, debug_flow_mode='color', pre_scale_target=True, metrics_save_path=None):
    """
    Main processing pipeline.
    """
    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ref_mri = cv2.imread(ref_mri_path, cv2.IMREAD_GRAYSCALE)
    ref_anime = cv2.imread(ref_anime_path)

    if ref_mri is None:
        print("Error: Could not read reference MRI image at {}".format(ref_mri_path))
        return

    if ref_anime is None:
        print("Error: Could not read reference anime image at {}".format(ref_anime_path))
        return

    target_frames = read_video_frames(target_video_path)
    if not target_frames:
        print("Error: Could not read frames from {}".format(target_video_path))
        return

    if pre_scale_target:
        target_size = (ref_anime.shape[1], ref_anime.shape[0])
        if target_frames[0].shape[1] != target_size[0] or target_frames[0].shape[0] != target_size[1]:
            target_frames = [
                cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
                for frame in target_frames
            ]

    target_dims = (target_frames[0].shape[0], target_frames[0].shape[1])

    lowest_mse = float('inf')
    best_transform = np.eye(2, 3, dtype=np.float32)
    best_frame_index = 0
    print("Processing '{}': Searching for the best frame using '{}' registration...".format(os.path.basename(target_video_path), registration_mode))
    for i, frame in enumerate(target_frames):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        transform, mse = get_registration_for_frame(ref_mri, gray_frame, mode=registration_mode)
        if (i + 1) % 10 == 0:
            print("  - Scanned frame {} | mse {}".format(i + 1, mse))
        if mse < lowest_mse:
            lowest_mse = mse
            best_transform = transform
            best_frame_index = i
    
    print("\nFound best match at frame {} with MSE {:.4f}\n".format(best_frame_index, lowest_mse))

    if metrics_save_path:
        import json as _json
        os.makedirs(os.path.dirname(os.path.abspath(metrics_save_path)), exist_ok=True)
        with open(metrics_save_path, "w") as _f:
            _json.dump(
                {"registration_error": float(lowest_mse), "anchor_index": int(best_frame_index)},
                _f,
                indent=2,
            )
        print("Registration metrics saved to {}".format(metrics_save_path))

    rigid_transform = best_transform
    
    h_orig, w_orig = target_dims
    h_ref, w_ref = ref_mri.shape

    sx = w_orig / w_ref
    sy = h_orig / h_ref

    rigid_transform[0, :] *= sx
    rigid_transform[1, :] *= sy
    
    if debug:
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        print("Debug mode enabled. Saving intermediate files to '{}'".format(debug_dir))
        
        warped_anime_for_debug = cv2.warpAffine(
            ref_anime,
            rigid_transform,
            (target_dims[1], target_dims[0]))
        best_frame_for_overlay = target_frames[best_frame_index]
        
        # Create a specific directory for these debug outputs
        debug_output_dir = "demo_items/debug_output"
        os.makedirs(debug_output_dir, exist_ok=True)

        # Save the requested debug images
        ref_frame_path = os.path.join(debug_output_dir, "debug_1_warped_reference_anime.png")
        target_frame_path = os.path.join(debug_output_dir, "debug_2_best_match_mri_target.png")
        overlay_path = os.path.join(debug_output_dir, "debug_3_overlay.png")
        cv2.imwrite(ref_frame_path, warped_anime_for_debug)
        cv2.imwrite(target_frame_path, best_frame_for_overlay)
        cv2.imwrite(overlay_path, cv2.addWeighted(best_frame_for_overlay, 0.6, warped_anime_for_debug, 0.4, 0))
        print(f"Saved debug images to '{debug_output_dir}'")
        #(target_dims[1], target_dims[0]))
        best_frame_for_overlay = target_frames[best_frame_index]
        blended_registration = cv2.addWeighted(best_frame_for_overlay, 0.6, warped_anime_for_debug, 0.4, 0)
        debug_path = os.path.join(debug_dir, "rigid_registration_overlay_{}.png".format(registration_mode))
        cv2.imwrite(debug_path, blended_registration)

    print("Calculating optical flow relative to best frame...")
    flow_fields = get_optical_flow(target_frames, best_frame_index)

    if debug:
        flow_video_path = os.path.join(debug_dir, "flow_visualization_{}.mp4".format(debug_flow_mode))
        flow_out = cv2.VideoWriter(flow_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (target_dims[1], target_dims[0]))
        print("Saving flow visualization debug video (mode: {})".format(debug_flow_mode))
        for i, flow in enumerate(flow_fields):
            frame = target_frames[i]
            if debug_flow_mode == 'arrow':
                vis_frame = visualize_flow_arrows(frame, flow)
            else: # default to 'color'
                flow_img = flow_to_image(flow)
                flow_img_bgr = cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR)
                vis_frame = cv2.addWeighted(frame, 0.6, flow_img_bgr, 0.4, 0)
            flow_out.write(vis_frame)
        flow_out.release()

    height, width, _ = ref_anime.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (target_dims[1], target_dims[0]), isColor=True)
    
    print("Warping anime frames and writing to video...")
    for i in range(len(flow_fields)):
        flow = flow_fields[i]
        warped_anime = warp_anime_frame(ref_anime, rigid_transform, flow, target_dims)
        if (i + 1) % 10 == 0:
            print("  - Processed frame {}".format(i + 1))
        out.write(warped_anime)

    out.release()
    print("Processed video saved to {}".format(output_video_path))

def process_video(target_video_path, output_dir, ref_mri_path, ref_anime_path, registration_mode='ecc', debug=False, debug_flow_mode='color', pre_scale_target=True, metrics_save_path=None):
    output_video_path = os.path.join(output_dir, os.path.basename(target_video_path))
    main(
        ref_mri_path,
        ref_anime_path,
        target_video_path,
        output_video_path,
        registration_mode,
        debug,
        debug_flow_mode,
        pre_scale_target,
        metrics_save_path=metrics_save_path,
    )

def process_single_image(target_image_path, output_dir, ref_mri_path, ref_anime_path):
    ref_mri = cv2.imread(ref_mri_path, cv2.IMREAD_GRAYSCALE)
    ref_anime = cv2.imread(ref_anime_path)
    target_image = cv2.imread(target_image_path)
    gray_frame = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    transform, _ = get_registration_for_frame(ref_mri, gray_frame, mode='ecc')
    warped_anime = cv2.warpAffine(
        ref_anime,
        transform,
        (target_image.shape[1], target_image.shape[0]))
    output_image_path = os.path.join(output_dir, os.path.basename(target_image_path))
    cv2.imwrite(output_image_path, warped_anime)
    print(f"Processed image saved to {output_image_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Animate MRI videos using a reference anime image.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the input video or image.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save the output animated video.")
    parser.add_argument('--ref-mri', type=str, default="data_sample/ref_mri.png", help="Path to the reference MRI image.")
    parser.add_argument('--ref-anime', type=str, default="data_sample/ref_anime.png", help="Path to the reference anime image.")
    parser.add_argument('--registration-mode', type=str, default='ecc', choices=['ecc', 'orb', 'sitk'], help="The registration mode to use.")
    parser.add_argument('--debug', action='store_true', help="Enable saving of intermediate debug files.")
    parser.add_argument('--debug-flow-mode', type=str, default='color', choices=['color', 'arrow'], help="The visualization mode for the optical flow debug video.")
    parser.add_argument('--single_image', action='store_true', help="Process a single image instead of a video.")
    parser.add_argument('--no-pre-scale-target', dest='pre_scale_target', action='store_false', help="Disable upscaling target video frames to match the reference anime resolution.")
    parser.set_defaults(pre_scale_target=True)

    args = parser.parse_args()

    if args.registration_mode == 'sitk' and not SITK_AVAILABLE:
        print("Error: SimpleITK is not installed. Please install it to use 'sitk' mode.")
        sys.exit(1)

    if args.single_image:
        process_single_image(args.input_dir, args.output_dir, args.ref_mri, args.ref_anime)
    else:
        process_video(
            args.input_dir,
            args.output_dir,
            args.ref_mri,
            args.ref_anime,
            args.registration_mode,
            args.debug,
            args.debug_flow_mode,
            args.pre_scale_target,
        )
