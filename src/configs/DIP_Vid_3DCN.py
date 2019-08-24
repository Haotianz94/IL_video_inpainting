from configs.base import cfg
cfg = cfg.copy()

# Dataset
cfg['batch_size'] = 5
cfg['batch_stride'] = [1]
cfg['batch_mode'] = 'random'
cfg['resize'] = (192, 384)

# Model
cfg['net_type_G'] = '3d'
cfg['filter_size_down'] = (3, 5, 5)
cfg['filter_size_up'] = (3, 3, 3)
cfg['filter_size_skip'] = (1, 1, 1)

# Loss
cfg['loss_weight'] = {'recon_image': 1, 'recon_flow': 0, 'consistency': 0, 'perceptual': 0}

# Optimize
cfg['train_mode'] = 'DIP-Vid-3DCN'
cfg['num_iter'] = 100
cfg['num_pass'] = 20
cfg['fine_tune'] = True
cfg['param_noise'] = False