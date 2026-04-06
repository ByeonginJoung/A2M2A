import argparse
import os
import subprocess

from run_pipeline import run_speech_to_mri, run_mri_to_anime


def extract_audio_from_video(video_file, output_audio, start=None, duration=None, sample_rate=44100, channels=2):
    """Extract audio from a video clip using ffmpeg."""
    cmd = ["ffmpeg", "-y"]
    if start is not None:
        cmd.extend(["-ss", str(start)])
    if duration is not None:
        cmd.extend(["-t", str(duration)])
    cmd.extend([
        "-i", video_file,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        output_audio,
    ])
    subprocess.run(cmd, check=True)
    return output_audio


def trim_video_segment(video_file, output_video, start=None, duration=None):
    """Trim the original video to align with the extracted audio segment."""
    if start is None and duration is None:
        return video_file
    cmd = ["ffmpeg", "-y"]
    if start is not None:
        cmd.extend(["-ss", str(start)])
    cmd.extend(["-i", video_file])
    if duration is not None:
        cmd.extend(["-t", str(duration)])
    cmd.extend(["-c", "copy", output_video])
    subprocess.run(cmd, check=True)
    return output_video


def concat_videos_side_by_side(left_path, middle_path, right_path, output_path):
    """Concatenate three videos horizontally using moviepy."""
    try:
        from moviepy.editor import VideoFileClip, clips_array
    except ImportError as exc:
        raise RuntimeError("moviepy is required for --concat_outputs but is not installed.") from exc

    video_clips = []
    processed = []
    stacked_clip = None
    try:
        video_clips = [VideoFileClip(left_path), VideoFileClip(middle_path), VideoFileClip(right_path)]
        min_duration = min(clip.duration for clip in video_clips)
        processed = [clip.subclip(0, min_duration) for clip in video_clips]

        target_height = processed[0].h
        processed[1] = processed[1].resize(height=target_height)
        processed[2] = processed[2].resize(height=target_height)

        stacked_clip = clips_array([[processed[0], processed[1], processed[2]]])
        if processed[0].audio is not None:
            stacked_clip = stacked_clip.set_audio(processed[0].audio)
        stacked_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    finally:
        if stacked_clip is not None:
            stacked_clip.close()
        for clip in processed:
            clip.close()
        for clip in video_clips:
            clip.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_file", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output_dir", type=str, default="pipeline_output", help="Directory to save the final output.")
    parser.add_argument("--log_dir", type=str, required=True, help="Path to the speech_to_2d_mri logs directory.")
    parser.add_argument("--use_prev_frame", action="store_true", help="Enable autoregressive generation for the MRI video.")
    parser.add_argument("--debug_anime", action="store_true", help="Save debug images for the MRI-to-Anime step.")
    parser.add_argument("--single_image", action="store_true", help="Process a single image instead of a video.")
    parser.add_argument("--no_pre_scale_target", dest="pre_scale_target", action="store_false", help="Disable upscaling target MRI frames before anime conversion.")
    parser.add_argument("--cut_video_start", type=float, default=None, help="Start time (in seconds) when extracting audio.")
    parser.add_argument("--cut_video_duration", type=float, default=None, help="Duration (in seconds) to keep when extracting audio.")
    parser.add_argument("--audio_sample_rate", type=int, default=44100, help="Sample rate for extracted audio.")
    parser.add_argument("--audio_channels", type=int, default=2, help="Number of channels for extracted audio.")
    parser.add_argument("--concat_outputs", action="store_true", help="Save a side-by-side video of original, MRI, and anime outputs.")
    parser.add_argument("--concat_output_path", type=str, default=None, help="Path to save the concatenated video output.")
    parser.add_argument("--save_metrics_path", type=str, default=None, help="If set, save registration metrics (anchor index, registration error) to this JSON file.")
    parser.add_argument("--audio_file", type=str, default=None, help="Path to a pre-existing audio file. When provided, audio extraction from --video_file is skipped (useful for MRI videos that have no embedded audio stream).")
    parser.set_defaults(pre_scale_target=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mri_output_dir = os.path.join(args.output_dir, "mri")
    anime_output_dir = os.path.join(args.output_dir, "anime")
    os.makedirs(mri_output_dir, exist_ok=True)
    os.makedirs(anime_output_dir, exist_ok=True)

    if args.audio_file:
        audio_source = args.audio_file
    else:
        extracted_audio_path = os.path.join(args.output_dir, "extracted_audio.wav")
        audio_source = extract_audio_from_video(
            args.video_file,
            extracted_audio_path,
            start=args.cut_video_start,
            duration=args.cut_video_duration,
            sample_rate=args.audio_sample_rate,
            channels=args.audio_channels,
        )

    trimmed_video_path = args.video_file
    if args.cut_video_start is not None or args.cut_video_duration is not None:
        trimmed_video_path = os.path.join(args.output_dir, "trimmed_input_video.mp4")
        trim_video_segment(
            args.video_file,
            trimmed_video_path,
            start=args.cut_video_start,
            duration=args.cut_video_duration,
        )

    run_speech_to_mri(audio_source, mri_output_dir, args.log_dir, args.use_prev_frame)

    generated_mri_video = None
    for f in os.listdir(mri_output_dir):
        if f.endswith(".mp4"):
            generated_mri_video = os.path.join(mri_output_dir, f)
            break

    if generated_mri_video:
        run_mri_to_anime(generated_mri_video, anime_output_dir, args.single_image, args.debug_anime, args.pre_scale_target, metrics_save_path=args.save_metrics_path)
    else:
        raise RuntimeError("Error: MRI video not found.")

    anime_video_path = os.path.join(anime_output_dir, os.path.basename(generated_mri_video))
    if args.concat_outputs:
        fname_stem = args.log_dir.split('/')[-1]
        if not os.path.exists(anime_video_path):
            raise RuntimeError("Anime video not found; cannot concatenate results.")
        video_stem = os.path.splitext(os.path.basename(args.video_file))[0]
        concat_output_path = args.concat_output_path or os.path.join(
            args.output_dir, f"comparison_{video_stem}_{fname_stem}.mp4"
        )
        concat_videos_side_by_side(trimmed_video_path, generated_mri_video, anime_video_path, concat_output_path)
