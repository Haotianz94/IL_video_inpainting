from configs.base import cfg
cfg = cfg.copy()

# Dataset
cfg['batch_size'] = 5
cfg['batch_stride'] = [1, 3, 5]
cfg['batch_mode'] = 'random'
cfg['resize'] = (192, 384)

# Model
cfg['net_type_G'] = '2d'

# Loss
cfg['loss_weight'] = {'recon_image': 1, 'recon_flow': 0.1, 'consistency': 1, 'perceptual': 0.01}

# Optimize
cfg['train_mode'] = 'DIP-Vid-Flow'
cfg['num_iter'] = 100
cfg['num_pass'] = 20
cfg['fine_tune'] = True
cfg['param_noise'] = False

# Result 
cfg['save_every_batch'] = 50
