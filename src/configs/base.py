import torch
import cv2
import logging

cfg = {}
# Dataset
cfg['video_path'] = 'data/bmx-trees.avi'
cfg['mask_path'] = 'data/bmx-trees_mask.avi'
cfg['batch_size'] = 5
cfg['batch_stride'] = [1]
cfg['batch_mode'] = 'random'
cfg['traverse_step'] = 1
cfg['frame_sum'] = 100
cfg['resize'] = None
cfg['interpolation'] = cv2.INTER_AREA
cfg['input_type'] = 'noise' # 'mesh_grid'
cfg['input_ratio'] = 0.1
cfg['dilation_iter'] = 0
cfg['input_noise_std'] = 0

# Model
cfg['net_type_G'] = '2d' # 3d
cfg['net_type_L'] = 'VGG16'
cfg['net_depth'] = 6
cfg['input_channel'] = 1
cfg['output_channel_img'] = 3
cfg['num_channels_down'] = [16, 32, 64, 128, 128, 128]
cfg['num_channels_up'] = [16, 32, 64, 128, 128, 128]
cfg['num_channels_skip'] = [4, 4, 4, 4, 4, 4]
cfg['filter_size_down'] = 5 
cfg['filter_size_up'] = 3
cfg['filter_size_skip'] = 1
cfg['use_skip'] = True
cfg['dtype'] = torch.cuda.FloatTensor

# Loss
cfg['loss_weight'] = {'recon_image': 1, 'recon_flow': 0, 'consistency': 0, 'perceptual': 0}
cfg['loss_recon'] = 'L2'
cfg['perceptual_layers'] = ['3', '8', '15']

# Optimize
cfg['train_mode'] = 'DIP-Vid-Flow'
cfg['baseline'] = True
cfg['LR'] = 1e-2
cfg['optimizer_G'] = 'Adam'
cfg['fine_tune'] = True
cfg['param_noise'] = True
cfg['num_iter'] = 100
cfg['num_pass'] = 20

# Result
cfg['save_every_iter'] = 100
cfg['save_every_pass'] = 1
cfg['plot'] = False
cfg['save'] = True
cfg['save_batch'] = False
cfg['res_dir'] = None

# Log
cfg['logging_level'] = logging.INFO # logging.DEBUG