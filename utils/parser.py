import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mri')
    parser.add_argument('--dataset_type', type=str, default='timit')
    parser.add_argument('--config_name', type=str, default='mri_base')
    parser.add_argument('--exp_name', type=str, default='exp0000')
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--sub_name', type=str, default='M1')
    parser.add_argument('--select_ckpt_idx', type=int, default=0)

    return parser.parse_args()
