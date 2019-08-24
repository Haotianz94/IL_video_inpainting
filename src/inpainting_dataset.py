import torch
import numpy as np
import os
from scipy import ndimage
import cv2
import random

from flow_estimator import FlowEstimator
from models.perceptual import LossNetwork
from utils import *


class InpaintingDataset(object):
    """
    Data loader for the input video 
    """    
    def __init__(self, cfg):
        self.cfg = cfg
        if not os.path.exists(self.cfg['video_path']):
            raise Exception("Input video not found: {}".format(self.cfg['video_path']))
        if not os.path.exists(self.cfg['mask_path']):
            raise Exception("Input mask not found: {}".format(self.cfg['mask_path']))
    
        cap = cv2.VideoCapture(self.cfg['video_path'])
        frame_sum_true = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if not self.cfg['resize'] is None:
            self.frame_H, self.frame_W = self.cfg['resize']
        self.frame_sum = self.cfg['frame_sum'] = min(self.cfg['frame_sum'], frame_sum_true)
        self.frame_size = self.cfg['frame_size'] = (self.frame_H , self.frame_W)
        self.batch_size = self.cfg['batch_size']
        self.batch_idx = 0
        self.batch_list_train = None

        if self.cfg['use_perceptual']:
            self.netL = LossNetwork(None, self.cfg['perceptual_layers'])
        
        self.init_frame_mask()
        self.init_input()
        self.init_flow()
        self.init_flow_mask()
        self.init_perceptual_mask()
        self.init_batch_list()


    def init_frame_mask(self):
        """
        Load input video and mask
        """
        self.image_all = []
        self.mask_all = []
        self.contour_all = []

        cap_video = cv2.VideoCapture(self.cfg['video_path'])
        cap_mask = cv2.VideoCapture(self.cfg['mask_path'])
        for fid in range(self.frame_sum):
            frame, mask = self.load_single_frame(cap_video, cap_mask)
            contour, hier = cv2.findContours(mask[0].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            self.image_all.append(frame)
            self.mask_all.append(mask)
            self.contour_all.append(contour)
        cap_video.release()
        cap_mask.release()
        self.image_all = np.array(self.image_all)
        self.mask_all = np.array(self.mask_all)


    def init_input(self):
        """
        Generate input noise map
        """
        input_noise = get_noise(self.frame_sum, self.cfg['input_channel'], 'noise', self.frame_size, var=self.cfg['input_ratio']).float().detach()
        input_noise = torch_to_np(input_noise) # N x C x H x W
        self.input_noise = input_noise


    def init_flow(self):
        """
        Estimate flow using PWC-Net
        """
        self.cfg['flow_value_max'] = self.flow_value_max = None
        if self.cfg['use_flow']:
            flow_estimator = FlowEstimator()
            print('Loading input video and estimating flow...')
            self.flow_all = { ft : [] for ft in self.cfg['flow_type']}
            self.flow_value_max = {}

            for bs in self.cfg['batch_stride']:
                f, b = 'f' + str(bs), 'b' + str(bs)
                for fid in range(0, self.frame_sum - bs):
                        frame_first = np_to_torch(self.image_all[fid]).clone()
                        frame_second = np_to_torch(self.image_all[fid + bs]).clone()
                        frame_first = frame_first.cuda()
                        frame_second = frame_second.cuda()
                        flow_f = flow_estimator.estimate_flow_pair(frame_first, frame_second).detach().cpu()
                        flow_b = flow_estimator.estimate_flow_pair(frame_second, frame_first).detach().cpu()
                        torch.cuda.empty_cache()

                        flow_f, flow_b = check_flow_occlusion(flow_f, flow_b)
                        self.flow_all[f].append(flow_f.numpy())
                        self.flow_all[b].append(flow_b.numpy())

            for bs in self.cfg['batch_stride']:
                f, b = 'f' + str(bs), 'b' + str(bs)
                self.flow_all[f] = np.array(self.flow_all[f] + [self.flow_all[f][0]] * bs, dtype=np.float32)
                self.flow_all[b] = np.array([self.flow_all[b][0]] * bs + self.flow_all[b], dtype=np.float32)
                self.flow_value_max[bs] = max(np.abs(self.flow_all[f]).max().astype('float'), \
                    np.abs(self.flow_all[b]).max().astype('float'))
            self.cfg['flow_value_max'] = self.flow_value_max


    def init_flow_mask(self):
        """
        Pre-compute warped mask and intersection of warped mask with original mask
        """
        if self.cfg['use_flow']:
            self.mask_warp_all = { ft : [] for ft in self.cfg['flow_type']}
            self.mask_flow_all = { ft : [] for ft in self.cfg['flow_type']}
            for bs in self.cfg['batch_stride']:
                    f, b = 'f' + str(bs), 'b' + str(bs)
                    # forward
                    mask_warpf, _ = warp_np(self.mask_all[bs:], self.flow_all[f][:-bs][:, :2, ...])
                    mask_warpf = (mask_warpf > 0).astype(np.float32)
                    mask_flowf = 1. - (1. - mask_warpf) * (1. - self.mask_all[:-bs])
                    self.mask_warp_all[f] = np.concatenate((mask_warpf, self.mask_all[-bs:]), 0)
                    self.mask_flow_all[f] = np.concatenate((mask_flowf, self.mask_all[-bs:]), 0)
                    # backward
                    mask_warpb, _ = warp_np(self.mask_all[:-bs], self.flow_all[b][bs:][:, :2, ...])
                    mask_warpb = (mask_warpb > 0).astype(np.float32)
                    mask_flowb = 1. - (1. - mask_warpb) * (1. - self.mask_all[bs:])
                    self.mask_warp_all[b] = np.concatenate((self.mask_all[:bs], mask_warpb), 0)
                    self.mask_flow_all[b] = np.concatenate((self.mask_all[:bs], mask_flowb), 0)


    def init_perceptual_mask(self):
        """
        Pre-compute shrinked mask for perceptual loss
        """
        if self.cfg['use_perceptual']:
            self.mask_per_all = []
            mask_per = self.netL(np_to_torch(self.mask_all))
            for i, mask in enumerate(mask_per):
                self.mask_per_all.append((mask.detach().numpy() > 0).astype(np.float32))

    
    def init_batch_list(self):
        """
        List all the possible batch permutations
        """
        if self.cfg['use_flow']:
            self.batch_list = []
            for flow_type in self.cfg['flow_type']:
                batch_stride = int(flow_type[1])
                for batch_idx in range(0, self.frame_sum - (self.batch_size - 1) * batch_stride, self.cfg['traverse_step']):
                    self.batch_list.append((batch_idx, batch_stride, [flow_type]))
            if self.cfg['batch_mode'] == 'random':
                random.shuffle(self.batch_list)
        else:
            for bs in self.cfg['batch_stride']:
                self.batch_list = self.batch_list = [(i, bs, []) for i in range(self.frame_sum - self.batch_size + 1)]
            if self.cfg['batch_mode'] == 'random':
                median = self.batch_list[len(self.batch_list) // 2]
                random.shuffle(self.batch_list)
                self.batch_list.remove(median)
                self.batch_list.append(median)


    def set_mode(self, mode):
        if mode == 'infer':
            self.batch_list_train = self.batch_list
            self.batch_list = [(i, 1, ['f1']) for i in range(self.frame_sum - self.batch_size + 1)]
        elif mode == 'train':
            if not self.batch_list_train is None:
                self.batch_list = self.batch_list_train
            else:
                self.init_batch_list()


    def next_batch(self):
        if len(self.batch_list) == 0:
            self.init_batch_list()
            return None
        else:
            (batch_idx, batch_stride, flow_type) = self.batch_list[0]
            self.batch_list = self.batch_list[1:]
            return self.get_batch_data(batch_idx, batch_stride, flow_type)


    def get_batch_data(self, batch_idx=0, batch_stride=1, flow_type=[]):
        """
        Collect batch data for centain batch 
        """
        cur_batch = range(batch_idx, batch_idx + self.batch_size*batch_stride, batch_stride)
        batch_data = {}
        input_batch, img_batch, mask_batch, contour_batch = [], [], [], []
        if self.cfg['use_flow']:
            flow_batch = { ft : [] for ft in flow_type}
            mask_flow_batch = { ft : [] for ft in flow_type}
            mask_warp_batch = { ft : [] for ft in flow_type}
        if self.cfg['use_perceptual']:
            mask_per_batch = [ [] for _ in self.cfg['perceptual_layers']]

        for i, fid in enumerate(cur_batch):
            input_batch.append(self.input_noise[fid])
            img_batch.append(self.image_all[fid])    
            mask_batch.append(self.mask_all[fid])            
            contour_batch.append(self.contour_all[fid])
            if self.cfg['use_flow']:
                for ft in flow_type:
                    flow_batch[ft].append(self.flow_all[ft][fid])
                    mask_flow_batch[ft].append(self.mask_flow_all[ft][fid])
                    mask_warp_batch[ft].append(self.mask_warp_all[ft][fid])
            if self.cfg['use_perceptual']:
                for l in range(len(self.cfg['perceptual_layers'])):
                    mask_per_batch[l].append(self.mask_per_all[l][fid])

       
        if self.cfg['use_flow']:
            for ft in flow_type:
                idx1, idx2 = (0, -1) if 'f' in ft else (1, self.batch_size)
                flow_batch[ft] = np.array(flow_batch[ft][idx1:idx2])
                mask_flow_batch[ft] = np.array(mask_flow_batch[ft][idx1:idx2])     
                mask_warp_batch[ft] = np.array(mask_warp_batch[ft][idx1:idx2])
        if self.cfg['use_perceptual']:
            for l in range(len(self.cfg['perceptual_layers'])):
                mask_per_batch[l] = np.array(mask_per_batch[l])     

        batch_data['cur_batch'] = cur_batch
        batch_data['batch_idx'] = batch_idx
        batch_data['batch_stride'] = batch_stride
        batch_data['input_batch'] = np.array(input_batch)
        batch_data['img_batch'] = np.array(img_batch)
        batch_data['mask_batch'] = np.array(mask_batch)
        batch_data['contour_batch'] = contour_batch
        if self.cfg['use_flow']:
            batch_data['flow_type'] = flow_type
            batch_data['flow_batch'] = flow_batch
            batch_data['mask_flow_batch'] = mask_flow_batch
            batch_data['mask_warp_batch'] = mask_warp_batch
            batch_data['flow_value_max'] = self.flow_value_max[batch_stride]
        if self.cfg['use_perceptual']:
            batch_data['mask_per_batch'] = mask_per_batch

        return batch_data


    def get_median_batch(self):
        return self.get_batch_data(int((self.cfg['frame_sum']) // 2), 1, ['f1'])


    def get_all_data(self):
        """
        Result a batch containing all the frames
        """
        batch_data = {}
        batch_data['input_batch'] = np.array(self.input_noise[:self.frame_sum])
        batch_data['img_batch'] = self.image_all
        batch_data['mask_batch'] = self.mask_all
        batch_data['contour_batch'] = self.contour_all
        if self.cfg['use_perceptual']:
            batch_data['mask_per_batch'] = self.mask_per_all
        if self.cfg['use_flow']:
            batch_data['flow_type'] = self.cfg['flow_type']
            batch_data['flow_batch'] = self.flow_all
            batch_data['mask_flow_batch'] = self.mask_flow_all
            batch_data['mask_warp_batch'] = self.mask_warp_all
        return batch_data


    def load_single_frame(self, cap_video, cap_mask):
        gt = self.load_image(cap_video, False, self.frame_size)
        mask = self.load_image(cap_mask, True, self.frame_size)
        if self.cfg['dilation_iter'] > 0:
            mask = ndimage.binary_dilation(mask > 0, iterations=self.cfg['dilation_iter']).astype(np.float32)
        return gt, mask

    
    def load_image(self, cap, is_mask, resize=None):
        _, img = cap.read()
        if not resize is None:
            img = self.crop_and_resize(img, resize)
        if is_mask:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
            img = (img > 127) * 255
            img = img.astype('uint8')
        img_convert = img.transpose(2, 0, 1)
        return img_convert.astype(np.float32) / 255
    
    
    def crop_and_resize(self, img, resize):
        """
        Crop and resize img, keeping relative ratio unchanged
        """
        h, w = img.shape[:2]
        source = 1. * h / w
        target = 1. * resize[0] / resize[1]
        if source > target:
            margin = int((h - w * target) // 2)
            img = img[margin:h-margin]
        elif source < target:
            margin = int((w - h / target) // 2)
            img = img[:, margin:w-margin]
        img = cv2.resize(img, (resize[1], resize[0]), interpolation=self.cfg['interpolation'])
        return img