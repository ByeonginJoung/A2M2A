import os
import logging
import subprocess

from datetime import datetime

def set_logger(args):
    
    args.log_dir = os.path.join('logs',
                                args.exp_name)

    # create dirs
    os.makedirs('logs', exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
         
    logger = logging.getLogger()      
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    now = datetime.now()
    time_now = now.strftime("%Y_%m_%d_%H_%M_%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(os.path.join(args.log_dir, f'{time_now}_mylog.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    # copy model folder to log folder
    os.makedirs(os.path.join(args.log_dir, 'backup_models'), exist_ok=True)
    args_copy = ['cp', '-r', 'models/', 'trainer/', 'dataset/', 'utils/', 'config/', f'{args.log_dir}/']
    subprocess.run(args_copy)
    return logger
