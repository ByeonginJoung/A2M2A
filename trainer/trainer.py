import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
import subprocess
import cv2
import numpy as np

from dataset.mri import MRI
from trainer.trainer_utils import build_optimizer_model, data_batchify

from utils.seed import set_seed
from utils.viz_utils import visualization

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

@torch.no_grad()
def test_eval(args, model, epoch_idx, val_loader, logger, res, data_stats, device):
    model.eval()

    mse_loss_list = list()

    #mgc_mean = torch.from_numpy(data_stats[0]).to(device)
    #mgc_std = torch.from_numpy(data_stats[1]).to(device)
    
    with torch.no_grad():
        for _, items in enumerate(tqdm(val_loader)):
            video = items[0].to(device)
            audio = items[1].to(device)

            H, W = video.shape[-2], video.shape[-1]
            
            new_video, new_audio = data_batchify(audio, video, args.data.lookback, args.data.fps_control_ratio)

            if args.model.use_deform:
                pred = model(new_audio.float(), video.view(-1, video.shape[-2], video.shape[-1])[0])
                pred = pred.view(-1, res[0], res[1])
            elif args.model.use_prev_frame:
                init_img = video.squeeze()[0]

                temp_vid_list = list()

                for proc_idx, temp_audio in enumerate(tqdm(new_audio)):
                    with torch.no_grad():
                        if proc_idx == 0:
                            in_aud = temp_audio.unsqueeze(0)
                            in_vid = init_img.unsqueeze(0)

                            temp_pred = model(in_aud, in_vid).view(H, W).cpu().detach().squeeze()
                        else:
                            in_aud = temp_audio.unsqueeze(0)
                            prev_vid = temp_vid_list[-1].unsqueeze(0).cuda()
                            
                            temp_pred = model(in_aud, prev_vid).view(H, W).cpu().detach().squeeze()
                        temp_vid_list.append(temp_pred)

                pred = torch.stack(temp_vid_list)
            else:
                pred = model(audio.float())
                pred = pred.view(-1, res[0], res[1])

            _mse_loss = torch.nn.functional.mse_loss(pred.cuda(), new_video)
            mse_loss_list.append(_mse_loss)

    mse_loss = sum(mse_loss_list) / len(mse_loss_list)

    # save file here
    if True:
        video_fname = items[-1][0]
        prediction = (pred.numpy() * 255).astype(np.uint8)
        process_list = ['cp', '-r', video_fname, 'demo_items/debug_eval_origin.avi']
        subprocess.run(process_list)

        output_file = f'demo_items/debug_eval_pred_{epoch_idx}.avi'
        
        if args.dataset_type == 'timit':
            fps = 23.18 / args.data.fps_control_ratio
        elif args.dataset_type == '75-speaker':
            fps = 83.28 / args.data.fps_control_ratio
        elif args.dataset_type == '75-speaker-word':
            fps = 83.28 / args.data.fps_control_ratio
        elif args.dataset_type == '75-speaker-multi':
            fps = 83.28 / args.data.fps_control_ratio
        elif args.dataset_type == '75-speaker-multi-word':
            fps = 83.28 / args.data.fps_control_ratio
        else:
            raise NotImplementedError
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec (you can use other codecs like 'XVID', 'MJPG', etc.)
        frame_size = (res[1], res[0])  # Frame size
        n_frames = pred.shape[0]

        # Create VideoWriter object
        out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

        # Convert and write frames to video
        for i in range(n_frames):
            # Convert single-channel grayscale to 3-channel BGR
            bgr_frame = cv2.cvtColor(prediction[i], cv2.COLOR_GRAY2BGR)
            out.write(bgr_frame)

        # Release the VideoWriter object
        out.release()
    
    print_info = f'total validation length: {len(mse_loss_list)}'
    print_info2 = f'mse loss for validation set: {mse_loss}'
    logger.info(print_info)
    logger.info(print_info2)

def run_trainer(args, logger):

    set_seed(args.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
    # load dataset
    dataset = MRI(args)
    val_dataset = MRI(args, val=True)
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             shuffle=True,
                                             drop_last=True)
    
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=1,
                                                 num_workers=1,
                                                 pin_memory=True,
                                                 shuffle=False,
                                                 drop_last=False)
    
    # set optimizer
    optimizer, scheduler, model, start_epoch, _ = build_optimizer_model(args,
                                                                        logger,
                                                                        dataset,
                                                                        device)
    model = model.to(device)
    
    data_stats = (dataset.mgc_mean, dataset.mgc_std)
    
    frame_H = dataset.frameHeight
    frame_W = dataset.frameWidth

    # TODO list to enhance the performance...
    # add perceptual loss such as SSIM
    # construct more robust AI model

    # this loss took too much time
    if args.use_ssimloss:
        ssim_func = SSIM(data_range=1.0, channel=1, win_size=args.ssimloss_window)

    print(f'start_epoch: {start_epoch}')
    
    for epoch_idx in tqdm(range(start_epoch, args.train_epoch+1)):
        model.train()

        for iteration, items in enumerate(dataloader):
            optimizer.zero_grad()

            video = items[0].to(device)
            mgc_lsp_coeff = items[1].to(device)

            _, B, H, W = video.shape
            
            sup_video, voice = data_batchify(mgc_lsp_coeff, video, args.data.lookback, args.data.fps_control_ratio)
            
            # run model
            # pred first image in initial
            if args.model.use_deform:
                pred = model(voice, video.view(B, H, W)[0])
            elif args.model.use_prev_frame:
                # since the voice and video were cut at data_batchify function,
                # the length of video was changed.
                # therefore change the code a little bit
                pred = model(voice, video.view(-1, H, W)[args.data.lookback-2:args.data.lookback-2+sup_video.shape[0]])
            else:
                pred = model(voice)
                
            # pred next image with given initial images
            loss_dict = dict()
            loss_dict['mse_loss'] = torch.nn.functional.mse_loss(pred, sup_video) * args.mseloss_weight
            
            if args.use_ssimloss:
                ssim_pred = pred.unsqueeze(2).view(-1, 1, frame_H, frame_W)
                ssim_video = sup_video.unsqueeze(2).view(-1, 1, frame_H, frame_W)
                
                loss_dict['ssim_loss'] = (1 - ssim_func(ssim_pred, ssim_video)) * args.ssimloss_weight

            if args.use_temporal_consistency:
                temp_cons = pred[:,1:] - pred[:,:-1]
                loss_dict['temp_cons_loss'] = torch.mean(temp_cons ** 2) * args.temporal_cons_weight

            loss = 0
            for key, value in loss_dict.items():
                loss = loss + value
            
            loss.backward()
            optimizer.step()
            
        if epoch_idx % args.epoch_print == 0 or epoch_idx == 1:
            init_viz = '\n Train {} | Epoch: {} | Iter: {} | '.format(args.exp_name, epoch_idx, iteration)
            loss_viz = 'loss: {:4f}'.format(loss.item())

            print_item = init_viz + loss_viz

            for key, value in loss_dict.items():
                temp_print_item = f' | {key}: {value.item():4f}'
                print_item += temp_print_item
            
            logger.info(print_item)

        if epoch_idx % args.epoch_eval == 0 and epoch_idx > 0:
            test_eval(args, model, epoch_idx, val_dataloader, logger, (frame_H, frame_W), data_stats, device)
                
        if epoch_idx % args.epoch_viz == 0 and epoch_idx > 0:
            visualization(args, pred, sup_video, epoch_idx, iteration)

        # Save checkpoint
        if epoch_idx % args.epoch_save == 0 and epoch_idx > 0:
            state = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch_idx,
                     'mgc_mean': data_stats[0] if args.data.feature_mode == 'mgclsp' else 0,
                     'mgc_std': data_stats[1] if args.data.feature_mode == 'mgclsp' else 0,
            }
            torch.save(state, os.path.join(args.log_dir, "ckpt_{:05d}".format(epoch_idx)))
