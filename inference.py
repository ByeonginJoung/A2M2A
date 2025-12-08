import os
import cv2
import argparse
import torch
import numpy as np
import torchaudio
import torchaudio.transforms as T
from trainer.trainer_utils import load_model
from config.utils import load_config
from utils.voice_converter import make_tts_like_ver2
from trainer.trainer_utils import data_batchify
from tqdm import tqdm
import subprocess

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    buf = np.empty((frameHeight, frameWidth, frameCount), np.dtype('float32'))
    fc = 0
    ret = True

    while (fc < frameCount and ret):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype('float32')
        if np.amax(frame) != 0:
            frame = (frame - np.amin(frame)) / np.amax(frame)
        buf[:, :, fc] = frame
        fc += 1
    cap.release()
    return buf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--config_name', type=str, default='mri_melspectogram_baseline_ver0004_multi')
    parser.add_argument('--dataset_type', type=str, default='75-speaker-multi')
    parser.add_argument('--exp_name', type=str, default='exp0000')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--select_ckpt_idx', type=int, default=-1)
    parser.add_argument('--use_prev_frame', action='store_true', help="Enable autoregressive generation using a reference frame.")
    
    args = parser.parse_args()
    
    new_args = load_config(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    if '75-speaker' in new_args.dataset_type:
        frame_H = 84
        frame_W = 84
    elif new_args.dataset_type == 'timit':
        frame_H = 68
        frame_W = 68
    else:
        raise NotImplementedError

    samplingFrequency = new_args.data.samplingFrequency
    frameLength = new_args.data.frameLength
    frameShift = new_args.data.frameShift

    output_wav = os.path.join(new_args.output_dir, 'temp.wav')
    make_tts_like_ver2(new_args.audio_file, output_wav)

    audio, sr = torchaudio.load(output_wav)

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=samplingFrequency,
        n_fft=frameLength,
        hop_length=frameShift,
        n_mels=new_args.model.in_feat
    )
    to_db = T.AmplitudeToDB()
    
    mel_spec = mel_spectrogram(audio)
    mel_spec_db = to_db(mel_spec).squeeze()

    audio_min = -80
    audio_max = 20

    audio = (mel_spec_db - audio_min) / (audio_max - audio_min)

    _, _, model, _, _ = load_model(new_args, new_args.model.in_feat, frame_H, frame_W, device)
    
    model.eval()
    model = model.to(device)
    
    _, dummy_audio = data_batchify(audio.T.unsqueeze(0).to(device), lookback=new_args.data.lookback, fps_control_ratio=new_args.data.fps_control_ratio)
    
    n_frames = dummy_audio.shape[0]
    frame_length = frameShift / samplingFrequency
    fps = 1 / frame_length / new_args.data.fps_control_ratio

    if new_args.model.use_prev_frame:
        print("Running in autoregressive mode.")
        # Load a hardcoded reference video to get a starting frame
        if '75-speaker' in new_args.dataset_type:
            ref_video_path = 'demo_items/sub051_2drt_07_grandfather1_r1_video.avi'
        elif new_args.dataset_type == 'timit':
            ref_video_path = 'demo_items/usctimit_mri_m1_011_015_withaudio.avi'
        else:
            raise NotImplementedError("No reference video specified for this dataset type.")
        
        random_video = load_video(ref_video_path)
        init_vid = random_video.transpose(-1, 0, 1)[10] # Use the 10th frame as the seed
        init_vid = torch.from_numpy(init_vid).to(device)
        
        temp_vid_list = []
        for proc_idx, temp_audio in enumerate(tqdm(dummy_audio, desc="Generating frames")):
            with torch.no_grad():
                prev_frame = init_vid.unsqueeze(0) if proc_idx == 0 else temp_vid_list[-1].unsqueeze(0).to(device)
                temp_pred = model(temp_audio.unsqueeze(0), prev_frame).view(frame_W, frame_H).cpu().detach()
                temp_vid_list.append(temp_pred)

        pred = torch.stack(temp_vid_list).numpy() * 255
    else:
        print("Running in direct generation mode.")
        with torch.no_grad():
            pred = model(dummy_audio.squeeze()).view(n_frames, frame_W, frame_H).cpu().detach().numpy() * 255
        
    pred = pred.astype(np.uint8)
    
    output_file = os.path.join(new_args.output_dir, 'output.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    frame_size = (frame_W, frame_H)

    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    for i in range(n_frames):
        bgr_frame = cv2.cvtColor(pred[i], cv2.COLOR_GRAY2BGR)
        out.write(bgr_frame)

    out.release()
    
    final_output = os.path.join(new_args.output_dir, os.path.basename(new_args.audio_file).replace('.wav', '.mp4'))
    
    # Embed audio
    subprocess.run(['ffmpeg', '-y', '-i', output_file, '-i', new_args.audio_file, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', final_output])

    print(f"Video saved as {final_output}")

if __name__ == '__main__':
    main()
