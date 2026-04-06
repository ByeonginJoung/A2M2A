import subprocess
import os
import argparse

def run_speech_to_mri(audio_file, output_dir, log_dir, use_prev_frame):
    """
    Runs the speech_to_2d_mri inference.py script.
    """
    cmd = [
        "python", "inference.py",
        "--audio_file", audio_file,
        "--output_dir", output_dir,
        "--log_dir", log_dir,
    ]
    if use_prev_frame:
        cmd.append("--use_prev_frame")
    subprocess.run(cmd, check=True)

def run_mri_to_anime(input_dir, output_dir, single_image, debug_anime, pre_scale_target=True, metrics_save_path=None):
    from main import process_single_image, process_video
    if single_image:
        process_single_image(input_dir, output_dir, "data_sample/ref_mri.png", "data_sample/ref_anime.png")
    else:
        process_video(
            input_dir,
            output_dir,
            "data_sample/ref_mri_0.png",
            "data_sample/ref_anime_0.png",
            debug=debug_anime,
            pre_scale_target=pre_scale_target,
            metrics_save_path=metrics_save_path,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_file", type=str, required=True, help="Path to the audio file.")
    parser.add_argument("--output_dir", type=str, default="pipeline_output", help="Directory to save the final output.")
    parser.add_argument("--log_dir", type=str, required=True, help="Path to the speech_to_2d_mri logs directory.")
    parser.add_argument("--use_prev_frame", action="store_true", help="Enable autoregressive generation for the MRI video.")
    parser.add_argument("--debug_anime", action="store_true", help="Save debug images for the MRI-to-Anime step.")
    parser.add_argument("--single_image", action="store_true", help="Process a single image instead of a video.")
    parser.add_argument("--no_pre_scale_target", dest="pre_scale_target", action="store_false", help="Disable upscaling target MRI frames before anime conversion.")
    parser.set_defaults(pre_scale_target=True)
    args = parser.parse_args()

    mri_output_dir = os.path.join(args.output_dir, "mri")
    anime_output_dir = os.path.join(args.output_dir, "anime")

    os.makedirs(mri_output_dir, exist_ok=True)
    os.makedirs(anime_output_dir, exist_ok=True)

    run_speech_to_mri(args.audio_file, mri_output_dir, args.log_dir, args.use_prev_frame)

    generated_mri_video = None
    for f in os.listdir(mri_output_dir):
        if f.endswith(".mp4"):
            generated_mri_video = os.path.join(mri_output_dir, f)
            break

    if generated_mri_video:
        run_mri_to_anime(generated_mri_video, anime_output_dir, args.single_image, args.debug_anime, args.pre_scale_target)
    else:
        print("Error: MRI video not found.")
