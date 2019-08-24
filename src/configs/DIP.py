from configs.base import cfg
cfg = cfg.copy()

# Dataset
cfg['batch_size'] = 1
cfg['batch_stride'] = [1]
cfg['batch_mode'] = 'seq'
cfg['resize'] = (192, 384)

# Model
cfg['net_type_G'] = '2d'

# Loss
cfg['loss_weight'] = {'recon_image': 1, 'recon_flow': 0, 'consistency': 0, 'perceptual': 0}

# Optimize
cfg['train_mode'] = 'DIP'
cfg['num_iter'] = 5000
cfg['num_pass'] = 1
cfg['fine_tune'] = False
cfg['param_noise'] = True