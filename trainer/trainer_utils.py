import os
import torch

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from models.speech_to_2d_mri import Speech2MRI2D

def build_optimizer_model(args, logger, dataset, device):

    return load_model(args,
                      args.model.in_feat,
                      dataset.frameHeight,
                      dataset.frameWidth,
                      device)

def load_model(args, n_input_feats, H, W, device):

    model = Speech2MRI2D(args,
                         n_input_feats,
                         H,
                         W)

    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)

    log_file_list = os.listdir(args.log_dir)

    ckpt_list = list()

    for fname in log_file_list:
        if 'ckpt' in fname:
            ckpt_list.append(fname)

    if len(ckpt_list) > 0:
        if args.select_ckpt_idx == 0:
            last_ckpt_fname = sorted(ckpt_list)[-1]
        else:
            last_ckpt_fname = sorted(ckpt_list)[args.select_ckpt_idx]

        ckpt_path = os.path.join(
            args.log_dir,
            last_ckpt_fname
            )

        state_dict = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(state_dict['model'])
        start_iter = state_dict['epoch']
        optimizer.load_state_dict(state_dict['optimizer'])

        print(f'load: {last_ckpt_fname} of {args.log_dir}')

        # Move optimizer state to the GPU
        for state in optimizer.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(device)
            elif isinstance(state, dict):
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        mgc_mean = state_dict['mgc_mean']
        mgc_std = state_dict['mgc_std']

        print(f'=======================================')
        print(f'=======================================')
        print(f'loaded model for epoch: {start_iter}')
        print(f'=======================================')
        print(f'=======================================')

    else:
        start_iter = 1
        mgc_mean = 0
        mgc_std = 0

    return optimizer, scheduler, model, start_iter, (mgc_mean, mgc_std)

def data_batchify(voice, video=None, lookback=10, fps_control_ratio=1):
    # video.shape: [B, L_vid, H, W]
    # voice.shape: [B, L_voi, C]

    # return batchfied video, batchfied audio, initial video

    # for general case, the B is 1, since there are video data

    # L_vid and L_voice can be different if the fps_control_ratio is not 1
    # for this case, the length of video and voice is not compatible,
    # we have to change this value properly

    if video is not None:
        _, L, H, W = video.shape
        video = video.view(L, H, W)
        new_video = video[lookback-1:]
    else:
        new_video = None

    _, L, C = voice.shape
    voice = voice.permute(1,0,2)

    idx1 = torch.arange(int(lookback*fps_control_ratio)).unsqueeze(0)
    idx2 = torch.arange(int((L-lookback*fps_control_ratio+1) // fps_control_ratio)).unsqueeze(1) * int(fps_control_ratio)
    idx = idx1 + idx2
    new_idx = idx.unsqueeze(-1).repeat(1, 1, C)
    
    new_voice = voice.squeeze().unsqueeze(0).expand(int(L-lookback*fps_control_ratio+1), -1, -1)
    new_audio = torch.gather(new_voice, 1, new_idx.cuda())

    # cut audio and video here again for shortest length
    # if there is video

    if video is not None:
        vd_length = min(video.shape[0], new_audio.shape[0])

        new_video = new_video[:vd_length]
        new_audio = new_audio[:vd_length]
        
    # debug:
    # new_voice[0] - voice.squeeze()[:10] ---> all 0
    return new_video, new_audio

