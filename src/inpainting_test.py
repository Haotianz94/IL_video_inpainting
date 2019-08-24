import torch
import torchvision
import numpy as np
import time
import os
import cv2
import logging

from inpainting_dataset import InpaintingDataset
from models.encoder_decoder_2d import EncoderDecoder2D
from models.encoder_decoder_3d import EncoderDecoder3D
from models.perceptual import LossNetwork
from utils import *


class InpaintingTest(object):
    """
    Internal learning framework
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.log = {}
        self.log['cfg'] = self.cfg.copy()
        for loss_name in self.cfg['loss_weight']:
            self.log['loss_' + loss_name + '_train'] = [ [] for _ in range(self.cfg['num_pass'])]
            self.log['loss_' + loss_name + '_infer'] = [ [] for _ in range(self.cfg['num_pass'])]
        self.data_loader = None
        self.netG = None
        self.optimizer_G = None
        
        # Pre-process cfg
        self.cfg['use_perceptual'] = self.cfg['loss_weight']['perceptual'] > 0
        self.cfg['use_flow'] = self.cfg['loss_weight']['recon_flow'] > 0

        if self.cfg['use_flow']:
            self.cfg['flow_type'] = []
            for bs in self.cfg['batch_stride']:
                self.cfg['flow_type'] += ['f' + str(bs), 'b' + str(bs)]
            self.cfg['flow_channel_map'] = {}
            channel_idx = self.cfg['output_channel_img'] 
            for ft in self.cfg['flow_type']:
                self.cfg['flow_channel_map'][ft] = (channel_idx, channel_idx+2)
                channel_idx += 2
        else:
            self.cfg['flow_type'] = None

        # Build result folder
        res_dir = self.cfg['res_dir']
        if not res_dir is None:
            res_dir = os.path.join(res_dir, os.path.basename(self.cfg['video_path']).split('.')[0])
            self.cfg['res_dir'] = res_dir
            if os.path.exists(res_dir):
                print("Warning: Video folder existed!")
            mkdir(res_dir)
            mkdir(os.path.join(res_dir, 'model'))
            for pass_idx in range(self.cfg['num_pass']):
                if (pass_idx + 1) % self.cfg['save_every_pass'] == 0:
                    res_dir = os.path.join(self.cfg['res_dir'], '{:03}'.format(pass_idx + 1))
                    mkdir(res_dir)
                    iter = self.cfg['save_every_iter']
                    while iter <= self.cfg['num_iter']:
                        build_dir(res_dir, '{:05}'.format(iter))
                        iter += self.cfg['save_every_iter']
                    build_dir(res_dir, 'final')
                    if self.cfg['train_mode'] == 'DIP':
                        build_dir(res_dir, 'best_nonhole')

        # Setup logging
        logging.basicConfig(level=self.cfg['logging_level'], format='%(message)s')
        self.logger = logging.getLogger(__name__)
        self.log_handler = None
        if not self.cfg['res_dir'] is None: 
            self.log_handler = logging.FileHandler(os.path.join(self.cfg['res_dir'], 'log.txt'))
            # self.log_handler.setLevel(logging.DEBUG)
            # formatter = logging.Formatter('%(message)s')
            # self.log_handler.setFormatter(formatter)
            self.logger.addHandler(self.log_handler)
        self.logger.info('========================================== Config ==========================================')
        for key in sorted(self.cfg):
            self.logger.info('[{}]: {}'.format(key, str(self.cfg[key])))


    def create_data_loader(self):
        self.logger.info('========================================== Dataset ==========================================')

        self.data_loader = InpaintingDataset(self.cfg)

        self.logger.info("[Video name]: {}".format(os.path.basename(self.cfg['video_path'])))
        self.logger.info("[Mask name]: {}".format(os.path.basename(self.cfg['mask_path'])))
        self.logger.info("[Frame sum]: {}".format(self.cfg['frame_sum']))
        self.logger.info("[Batch size]: {}".format(self.cfg['batch_size']))
        self.logger.info("[Frame size]: {}".format(self.cfg['frame_size']))
        self.logger.info("[Flow type]: {}".format(self.cfg['flow_type']))
        self.logger.info("[Flow_value_max]: {}".format(self.cfg['flow_value_max']))

        self.log['input_noise'] = self.data_loader.input_noise


    def visualize_single_batch(self):
        """
        Randomly visualize one batch data
        """
        batch_data = self.data_loader.next_batch()
        input_batch = batch_data['input_batch']
        img_batch = batch_data['img_batch']
        mask_batch = batch_data['mask_batch']
        nonhole_batch = img_batch * (1 - mask_batch)
        
        if self.cfg['batch_size'] == 1:
            plot_image_grid(np.concatenate((img_batch, nonhole_batch), 0), 2, padding=3, factor=10)
        else:
            plot_image_grid(np.concatenate((img_batch, nonhole_batch), 0), self.cfg['batch_size'], padding=3, factor=15)
            
        
    def create_model(self):
        if not self.cfg['use_skip']:
            num_channels_skip = [0] * self.cfg['net_depth']
        
        input_channel = self.cfg['input_channel']
        if self.cfg['use_flow']:
            output_channel_flow = len(self.cfg['flow_type']) * 2
            output_channel = self.cfg['output_channel_img'] + output_channel_flow
        else:
            output_channel = self.cfg['output_channel_img'] 

        if self.cfg['net_type_G'] == '2d':
            self.netG = EncoderDecoder2D(input_channel, output_channel, 
                   self.cfg['num_channels_down'][:self.cfg['net_depth']],
                   self.cfg['num_channels_up'][:self.cfg['net_depth']],
                   self.cfg['num_channels_skip'][:self.cfg['net_depth']],  
                   self.cfg['filter_size_down'], self.cfg['filter_size_up'], self.cfg['filter_size_skip'],
                   upsample_mode='nearest', downsample_mode='stride',
                   need1x1_up=True, need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        elif self.cfg['net_type_G'] == '3d':
            self.netG = EncoderDecoder3D(input_channel, output_channel, 
                   self.cfg['num_channels_down'][:self.cfg['net_depth']],
                   self.cfg['num_channels_up'][:self.cfg['net_depth']],
                   self.cfg['num_channels_skip'][:self.cfg['net_depth']],  
                   self.cfg['filter_size_down'], self.cfg['filter_size_up'], self.cfg['filter_size_skip'],
                   upsample_mode='nearest', downsample_mode='stride',
                   need1x1_up=True, need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        else:
            raise Exception("Network not defined!")
        self.netG = self.netG.type(self.cfg['dtype'])
       
        if self.cfg['use_perceptual']:
            if self.cfg['net_type_L'] == 'VGG16':
                vgg_model = torchvision.models.vgg16(pretrained=True).type(self.cfg['dtype'])
                vgg_model.eval()
                vgg_model.requires_grad = False
                self.netL = LossNetwork(vgg_model)

        self.logger.info('========================================== Network ==========================================')
        self.logger.info(self.netG)
        self.logger.info("Total number of parameters: {}".format(get_model_num_parameters(self.netG)))


    def create_loss_function(self):
        if self.cfg['loss_recon'] == 'L1':
            self.criterion_recon = torch.nn.L1Loss().type(self.cfg['dtype'])
        elif self.cfg['loss_recon'] == 'L2':
            self.criterion_recon = torch.nn.MSELoss().type(self.cfg['dtype'])
        self.criterion_MSE = torch.nn.MSELoss().type(self.cfg['dtype'])


    def create_optimizer(self):
        if self.cfg['optimizer_G'] == 'Adam':
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.cfg['LR'])
        elif self.cfg['optimizer_G'] == 'SGD':
            self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=self.cfg['LR'])

    
    def prepare_input(self, input_batch):
        """
        Prepare input noise map based on network type (2D/3D)
        """
        input_tensor = np_to_torch(input_batch).type(self.cfg['dtype']) # N x C x H x W
        if self.cfg['net_type_G'] == '2d':
            return input_tensor # N x C x H x W
        elif self.cfg['net_type_G'] == '3d':
            return input_tensor.transpose(0, 1).unsqueeze(0) # 1 x C x N x H x W

    
    def train(self):
        """
        Main function for internal learning
        """
        self.start_time = time.time()
        self.data_loader.init_batch_list()

        self.logger.info('========================================== Training ==========================================')
        if self.cfg['train_mode'] == 'DIP-Vid-Flow':
            self.train_with_flow()
        else:
            self.train_baseline()

        # Save log
        if not self.cfg['res_dir'] is None:
            torch.save(self.log, os.path.join(self.cfg['res_dir'], 'log.tar'))

        # Report training time
        running_time = time.time() - self.start_time
        self.logger.info("Training finished! Running time: {}s".format(running_time))

        # Release log file
        self.logger.removeHandler(self.log_handler)


    def train_baseline(self):
        """
        Training procedure for all baselines
        """
        for pass_idx in range(self.cfg['num_pass']):
            while True:
                # Get batch data
                batch_data = self.data_loader.next_batch()
                if batch_data is None:
                    break
                batch_idx = batch_data['batch_idx']

                self.logger.info('Pass {},  Batch {}'.format(pass_idx + 1, batch_idx))

                # Start train
                self.train_batch(pass_idx, batch_idx, batch_data)

                if self.cfg['train_mode'] == 'DIP':
                    self.create_model()
                    self.create_optimizer()

            # Start infer
            if (pass_idx + 1) % self.cfg['save_every_pass'] == 0 and self.cfg['train_mode'] != 'DIP':
                inferred_result = self.infer(pass_idx)
            
                self.logger.info("Saving latest model at pass {}".format(pass_idx + 1))
                if not self.cfg['res_dir'] is None:
                    # Save model and log
                    checkpoint_G = os.path.join(self.cfg['res_dir'], 'model', '{:03}.tar'.format(pass_idx + 1))
                    torch.save(self.netG.state_dict(), checkpoint_G)
                    torch.save(self.log, os.path.join(self.cfg['res_dir'], 'log.tar'))
            
            self.logger.info("Running time: {}s".format(time.time() - self.start_time))




    def train_with_flow(self):
        """
        Training procedure for DIP-Vid-Flow
        """
        pass_idx = 0
        batch_count = 0
        while pass_idx < self.cfg['num_pass']:
            # Get batch data
            batch_data = self.data_loader.next_batch()
            if batch_data is None:
                continue
            batch_idx = batch_data['batch_idx']

            self.logger.info('Pass: {}, Batch: {}, Flow: {}'.format(pass_idx, batch_idx, str(batch_data['flow_type'])))

            # Start train
            self.train_batch(pass_idx, batch_idx, batch_data)
            batch_count += 1

            # Start infer
            if batch_count % self.cfg['save_every_batch'] == 0:
                batch_data = self.data_loader.get_median_batch()
                batch_idx = batch_data['batch_idx']
                self.logger.info('Train the median batch before inferring\nPass: {}, Batch: {}, Flow: {}'.format(pass_idx, batch_idx, str(batch_data['flow_type'])))
                self.train_batch(pass_idx, batch_idx, batch_data)

                self.infer(pass_idx)
                
                # Save model and log
                self.logger.info("Running time: {}s".format(time.time() - self.start_time))
                if not self.cfg['res_dir'] is None:
                    checkpoint_G = os.path.join(self.cfg['res_dir'], 'model', '{:03}.tar'.format(pass_idx + 1))
                    torch.save(self.netG.state_dict(), checkpoint_G)
                    torch.save(self.log, os.path.join(self.cfg['res_dir'], 'log.tar'))
                pass_idx += 1

    
    def infer(self, pass_idx):
        """
        Run inferrance with trained model to collect all inpainted frames 
        """
        self.logger.info('Pass {} infer start...'.format(pass_idx))
        self.data_loader.set_mode('infer')

        inferred_result = np.empty((self.cfg['frame_sum'], self.cfg['output_channel_img'], self.cfg['frame_size'][0], self.cfg['frame_size'][1]), dtype=np.float32)
        while True:
            batch_data = self.data_loader.next_batch()
            if batch_data is None:
                break
            batch_idx = batch_data['batch_idx']
            self.infer_batch(pass_idx, batch_idx, batch_data, inferred_result)
            
        self.data_loader.set_mode('train')
        return inferred_result


    def train_batch(self, pass_idx, batch_idx, batch_data):
        """
        Train the given batch for `num_iter` iterations 
        """
        for loss_name in self.cfg['loss_weight']:
            self.log['loss_' + loss_name + '_train'][pass_idx].append([])
        best_loss_recon_image = 1e9
        best_iter = 0
        best_nonhole_batch = None
        batch_data['pass_idx'] = pass_idx
        batch_data['train'] = True
        
        # Optimize
        for iter_idx in range(self.cfg['num_iter']):
            if self.cfg['param_noise']:
                for n in [x for x in self.netG.parameters() if len(x.size()) == 4]:
                    n = n + n.detach().clone().normal_() * n.std() / 50
            
            # Forward
            loss = self.optimize_params(batch_data)
            
            # Update
            for loss_name in self.cfg['loss_weight']:
                self.log['loss_' + loss_name + '_train'][pass_idx][-1].append(loss[loss_name].item())
            if loss['recon_image'].item() < best_loss_recon_image:
                best_loss_recon_image = loss['recon_image'].item()
                best_nonhole_batch = batch_data['out_img_batch']
                best_iter = iter_idx
            
            log_str = 'Iteration {:05}'.format(iter_idx)
            for loss_name in sorted(self.cfg['loss_weight']):
                if self.cfg['loss_weight'][loss_name] != 0:
                    log_str += '  ' + loss_name + ' {:f}'.format(loss[loss_name].item())
            self.logger.info(log_str)
            
            # Plot and save
            if (pass_idx + 1) % self.cfg['save_every_pass'] == 0 and (iter_idx + 1) % self.cfg['save_every_iter'] == 0:
                self.plot_and_save(batch_idx, batch_data, '{:03}/{:05}'.format(pass_idx + 1, iter_idx + 1))

        log_str = 'Best at iteration {:05}, recon_image loss {:f}'.format(best_iter, best_loss_recon_image)
        self.logger.info(log_str)

        if self.cfg['train_mode'] == 'DIP':
            # For DIP, save the result with lowest loss on nonhole region as final result
            batch_data['out_img_batch'] = best_nonhole_batch
            self.plot_and_save(batch_idx, batch_data, '001/best_nonhole')

    
    def infer_batch(self, pass_idx, batch_idx, batch_data, inferred_result):
        """
        Run inferrance for the given batch
        """
        # Forward pass
        batch_data['pass_idx'] = pass_idx
        batch_data['train'] = False
        loss = self.optimize_params(batch_data)

        # Update
        for loss_name in self.cfg['loss_weight']:
            self.log['loss_' + loss_name + '_infer'][pass_idx].append(loss[loss_name].item())

        # Save inferred result
        for i, img in enumerate(batch_data['out_img_batch'][0]):
            if batch_idx < batch_data['batch_stride'] or i >= self.cfg['batch_size'] // 2: 
                inferred_result[batch_idx + i * batch_data['batch_stride']] = img

        log_str = 'Batch {:05}'.format(batch_idx)
        for loss_name in sorted(self.cfg['loss_weight']):
            if self.cfg['loss_weight'][loss_name] != 0:
                log_str += '  ' + loss_name + ' {:f}'.format(loss[loss_name].item())
        self.logger.info(log_str)            

        # Plot and save
        if (self.cfg['plot'] or self.cfg['save']) and (pass_idx + 1) % self.cfg['save_every_pass'] == 0:   
            self.plot_and_save(batch_idx, batch_data, '{:03}/final'.format(pass_idx + 1))


    def optimize_params(self, batch_data):
        """
        Calculate loss and back-propagate the loss
        """
        pass_idx = batch_data['pass_idx']
        batch_idx = batch_data['batch_idx']
        net_input = self.prepare_input(batch_data['input_batch'])
        img_tensor = np_to_torch(batch_data['img_batch']).type(self.cfg['dtype'])
        mask_tensor = np_to_torch(batch_data['mask_batch']).type(self.cfg['dtype'])
        
        if self.cfg['use_flow']:
            flow_tensor, mask_flow_tensor, mask_warp_tensor = {}, {}, {}
            for ft in batch_data['flow_type']:
                flow_tensor[ft] = np_to_torch(batch_data['flow_batch'][ft]).type(self.cfg['dtype'])
                mask_flow_tensor[ft] = np_to_torch(batch_data['mask_flow_batch'][ft]).type(self.cfg['dtype'])
                mask_warp_tensor[ft] = np_to_torch(batch_data['mask_warp_batch'][ft]).type(self.cfg['dtype'])
            flow_value_max = batch_data['flow_value_max']

        if self.cfg['use_perceptual']:
            mask_per_tensor = []
            for mask in batch_data['mask_per_batch']:
                mask = np_to_torch(mask).type(self.cfg['dtype'])
                mask_per_tensor.append(mask)

        # Forward
        net_output = self.netG(net_input)
        torch.cuda.empty_cache()

        # Collect image/flow from network output
        if self.cfg['net_type_G'] == '2d':
            out_img_tensor = net_output[:, :self.cfg['output_channel_img'], ...] # N x 3 x H x W
            if self.cfg['use_flow']:
                out_flow_tensor = {}
                for ft in batch_data['flow_type']:
                    channel_idx1, channel_idx2 = self.cfg['flow_channel_map'][ft]
                    flow_idx1, flow_idx2 = (0, -1) if 'f' in ft else (1, self.cfg['batch_size'])
                    out_flow_tensor[ft] = net_output[flow_idx1:flow_idx2, channel_idx1:channel_idx2, ...]

        elif self.cfg['net_type_G'] == '3d':
            out_img_tensor = net_output.squeeze(0)[:self.cfg['output_channel_img']].transpose(0, 1) # N x 3 x H x W
            if self.cfg['use_flow']:
                out_flow_tensor = {}
                for ft in batch_data['flow_type']:
                    channel_idx1, channel_idx2 = self.cfg['flow_channel_map'][ft]
                    flow_idx1, flow_idx2 = (0, -1) if 'f' in ft else (1, self.cfg['batch_size'])
                    out_flow_tensor[ft] = net_output.squeeze(0) \
                    [channel_idx1:channel_idx2, flow_idx1:flow_idx2, ...].transpose(0, 1) # N-1 x 2 x H x W

        # Compute loss
        loss = {}
        for loss_name in self.cfg['loss_weight']:
            loss[loss_name] = torch.zeros([]).float().cuda().detach()

        self.optimizer_G.zero_grad()
        
        # Image reconstruction loss
        if self.cfg['loss_weight']['recon_image'] != 0:
                loss['recon_image'] += self.criterion_recon(
                    out_img_tensor * (1. - mask_tensor), \
                    img_tensor * (1. - mask_tensor))
        
        # Flow reconstruction loss
        if self.cfg['loss_weight']['recon_flow'] != 0:
            for ft in batch_data['flow_type']:
                mask_flow_inv = (1. - mask_flow_tensor[ft]) * flow_tensor[ft][:, 2:3, ...]
                loss['recon_flow'] += self.criterion_recon(out_flow_tensor[ft] * mask_flow_inv, \
                    flow_tensor[ft][:, :2, ...]  * mask_flow_inv / flow_value_max)

        # Consistency loss
        if self.cfg['loss_weight']['consistency'] != 0:
            warped_img, warped_diff = {}, {}
            for ft in batch_data['flow_type']:
                idx1, idx2 = (0, -1) if 'f' in ft else (1, self.cfg['batch_size'])
                idx_inv1, idx_inv2 = (1, self.cfg['batch_size']) if 'f' in ft else (0, -1)
                out_img = out_img_tensor[idx_inv1:idx_inv2]
                out_flow = out_flow_tensor[ft]
                warped_img[ft], flowmask = warp_torch(out_img, out_flow * flow_value_max)
                mask = mask_flow_tensor[ft] * flowmask.detach()
                loss['consistency'] += self.criterion_recon(
                    warped_img[ft] * mask,
                    out_img_tensor[idx1:idx2].detach() * mask)
                torch.cuda.empty_cache()

        # Perceptual loss            
        if self.cfg['use_perceptual']:
            feature_src = self.netL(out_img_tensor)
            feature_dst = self.netL(img_tensor)
            for i, mask in enumerate(mask_per_tensor):
                loss['perceptual'] += self.criterion_MSE(
                    feature_src[i] * (1. - mask), feature_dst[i].detach() * (1. - mask))
            torch.cuda.empty_cache()

        # Back-propagation
        running_loss  = 0
        for loss_name, weight in self.cfg['loss_weight'].items():
            if weight != 0:
                running_loss = running_loss + weight * loss[loss_name]
        
        if batch_data['train']:        
            running_loss.backward()
            self.optimizer_G.step()
            torch.cuda.empty_cache()

        # Save generated image/flow
        batch_data['out_img_batch'] = torch_to_np(out_img_tensor)
        if self.cfg['use_flow']:
            out_flow_batch = {}
            for ft in batch_data['flow_type']:
                out_flow_batch[ft] = torch_to_np(out_flow_tensor[ft] * flow_value_max)
            batch_data['out_flow_batch'] = out_flow_batch
        return loss 
            
    
    def plot_and_save(self, batch_idx, batch_data, subpath):
        """
        Plot/save intermediate results
        """
        def save(imgs, subpath, subsubpath):
            res_dir = os.path.join(self.cfg['res_dir'], subpath, subsubpath)
            for i, img in enumerate(imgs):
                if img is None:
                    continue
                fid = batch_idx + i * batch_data['batch_stride']
                batch_path = os.path.join(res_dir, 'batch', '{:03}_{:03}.png'.format(batch_idx, fid))
                sequence_path = os.path.join(res_dir, 'sequence', '{:03}.png'.format(fid))
                if self.cfg['save_batch']:
                    cv2.imwrite(batch_path, np_to_cv2(img))
                if batch_idx < batch_data['batch_stride'] or i >= self.cfg['batch_size'] // 2:
                    cv2.imwrite(sequence_path, np_to_cv2(img))

        # Load batch data
        input_batch = batch_data['input_batch'] / self.cfg['input_ratio']
        img_batch = batch_data['img_batch']
        mask_batch = batch_data['mask_batch']
        contour_batch = batch_data['contour_batch']
        out_img_batch = batch_data['out_img_batch'].copy()
        stitch_batch = img_batch * (1 - mask_batch) + out_img_batch * mask_batch

        # Draw mask boundary
        for i in range(self.cfg['batch_size']):
            for con in contour_batch[i]:
                for pt in con:
                    x, y = pt[0]
                    out_img_batch[i][:, y, x] = [0, 0, 1]
                        
        # Plot in jupyter
        if self.cfg['plot']:
            if self.cfg['batch_size'] == 1:
                plot_image_grid(out_img_batch, 1, factor=10)
            else:
                plot_image_grid(out_img_batch, self.cfg['batch_size'], padding=3, factor=15)

        # Save images to disk
        if not self.cfg['res_dir'] is None and self.cfg['save']:
            save(stitch_batch, subpath, 'stitch')
            save(out_img_batch, subpath, 'full_with_boundary')
            save(batch_data['out_img_batch'], subpath, 'full')