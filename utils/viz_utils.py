import os
import torch

import numpy as np

import matplotlib.pyplot as plt

def visualization(args,
                  pred,
                  video,
                  epoch,
                  iters):

    save_path = os.path.join(args.log_dir, 'viz_imgs')
    os.makedirs(save_path, exist_ok=True)

    save_img_path = os.path.join(args.log_dir, 'viz_imgs', f'{epoch:03d}_{iters:05d}.png')
    
    plt.figure()
    plt.tight_layout()
    plt.subplot(121)
    plt.imshow(pred[0].cpu().detach().numpy())
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(video[0].cpu().detach().numpy())
    plt.axis('off')
    plt.savefig(save_img_path)
    plt.clf()
