from easydict import EasyDict as edict

def get_args():
    args_edict = edict()
    args_edict.dataset_dir = '/hdd4tb_00/dataset/mri_data'
    
    args_edict.seed = 1234

    args_edict.train_epoch = 1000
    args_edict.epoch_print = 20
    args_edict.epoch_viz = 100
    args_edict.epoch_save = 100
    args_edict.epoch_eval = 100
    
    args_edict.batch_size = 1
    args_edict.num_workers = 1

    args_edict.lr = 0.0001

    args_edict.model = edict()
    args_edict.model.n_feats = 1024
    args_edict.model.use_lstm = True
    args_edict.model.use_transformer = False
    args_edict.model.n_head = 8
    args_edict.model.use_bn = False
    args_edict.model.use_dropout = True
    args_edict.model.residual = True
    args_edict.model.use_deform = False
    args_edict.model.use_prev_frame = True

    args_edict.model.in_feat = 128 # for mgclsp, generally 25, for raw, 534, for melspectogram, 80
    
    args_edict.mseloss_weight = 1.
    
    args_edict.use_ssimloss = True
    args_edict.ssimloss_window = 9
    args_edict.ssimloss_weight = 0.1

    args_edict.use_temporal_consistency = False
    args_edict.temporal_cons_weight = 0.05

    args_edict.data = edict()
    args_edict.data.feature_mode = 'melspectogram'
    args_edict.data.order = 24
    args_edict.data.alpha = 0.42
    args_edict.data.stage = 3
    args_edict.data.lookback = 10
    args_edict.data.samplingFrequency = 44100
    args_edict.data.frameLength = 2048
    args_edict.data.frameShift = 533
    args_edict.data.fps_control_ratio = 6. # strongly recommended usage for this value as int type
    args_edict.data.cut_rand_initial = True

    args_edict.data.augmentation = edict()
    args_edict.data.augmentation.use_augment = True
    args_edict.data.augmentation.add_gaussian = True
    args_edict.data.augmentation.add_gaussian_mean = 0.0
    args_edict.data.augmentation.add_gaussian_std = 1
    args_edict.data.augmentation.add_spectral_scaling = False   # this is for mgclsp
    args_edict.data.augmentation.spectral_scaling_factor = 0.1
    args_edict.data.augmentation.add_time_stretch_mgc = False   # this is for mgclsp
    args_edict.data.augmentation.add_time_stretch = False
    args_edict.data.augmentation.time_stretch_factor=1.1
    args_edict.data.augmentation.add_random_erasing = False
    args_edict.data.augmentation.random_erasing_prob=0.5
    args_edict.data.augmentation.random_erasing_size=0.1
        
    return args_edict
