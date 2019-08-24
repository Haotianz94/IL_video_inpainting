######################################################################
# Code modified from https://github.com/DmitryUlyanov/deep-image-prior
######################################################################

import torch
import torchvision
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


######################################################################
# Network input
######################################################################

def fill_noise(x, noise_type):
    """
    Fill tensor `x` with noise of type `noise_type`.
    """
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False


def get_noise(batch_size, input_depth, method, spatial_size, noise_type='u', var=1./10):
    """
    Return a pytorch.Tensor of size (`batch_size` x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [batch_size, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid':
        assert batch_size == 1
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1), np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input


######################################################################
# Flow related
######################################################################

def warp_torch(x, flo):
    """
    Backward warp an image tensor (im2) to im1, according to the optical flow from im1 to im2

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # Mesh grid 
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid_ori = grid + flo
    vgrid = torch.zeros_like(vgrid_ori)

    # Scale grid to [-1,1] 
    vgrid[:, 0, :, :] = 2.0 * vgrid_ori[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid_ori[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)        
    output = torch.nn.functional.grid_sample(x, vgrid)
    mask = torch.ones(x.size())
    if x.is_cuda:
        mask = mask.cuda()
    mask = torch.nn.functional.grid_sample(mask, vgrid)
    
    mask[mask < 0.999] = 0
    mask[mask > 0] = 1
    
    return output, mask

                                         
def warp_np(x, flo):
    """
    Backward warp an image numpy array (im2) to im1, according to the optical flow from im1 to im2

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    if x.ndim != 4:
        assert(x.ndim == 3)
        # Add one dimention for single image
        x = x[None, ...]
        flo = flo[None, ...]
        add_dim = True
    else:
        add_dim = False

    output, mask = warp_torch(np_to_torch(x), np_to_torch(flo))
    if add_dim:
        return output.numpy()[0], mask.numpy()[0]
    else:
        return output.numpy(), mask.numpy()


def check_flow_occlusion(flow_f, flow_b):
    """
    Compute occlusion map through forward/backward flow consistency check
    """
    def get_occlusion(flow1, flow2):
        grid_flow = grid + flow1
        grid_flow[0, :, :] = 2.0 * grid_flow[0, :, :] / max(W - 1, 1) - 1.0
        grid_flow[1, :, :] = 2.0 * grid_flow[1, :, :] / max(H - 1, 1) - 1.0
        grid_flow = grid_flow.permute(1, 2, 0)        
        flow2_inter = torch.nn.functional.grid_sample(flow2[None, ...], grid_flow[None, ...])[0]
        score = torch.exp(- torch.sum((flow1 + flow2_inter) ** 2, dim=0) / 2.)
        occlusion = (score > 0.5)
        return occlusion[None, ...].float()

    C, H, W = flow_f.size()
    # Mesh grid 
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, H, W)
    yy = yy.view(1, H, W)
    grid = torch.cat((xx, yy), 0).float()

    occlusion_f = get_occlusion(flow_f, flow_b)
    occlusion_b = get_occlusion(flow_b, flow_f)
    flow_f = torch.cat((flow_f, occlusion_f), 0)
    flow_b = torch.cat((flow_b, occlusion_b), 0)

    return flow_f, flow_b


######################################################################
# Visualization
######################################################################

def get_image_grid(images_np, nrow=8, padding=2):
    """
    Create a grid from a list of images by concatenating them.
    """
    images_torch = [np_to_torch(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow, padding)
    
    return torch_grid.numpy()


def plot_image_grid(images_np, nrow=8, padding=2, factor=1, interpolation='lanczos'):
    """
    Layout images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [np_cvt_color(x) if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow, padding)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    plt.axis('off')

    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    
    plt.show()
    
    return grid


######################################################################
# Data type transform
######################################################################

def np_cvt_color(img_np):
    """
    Convert image from BGR/RGB to RGb/BGR
    From B x C x W x H  to B x C x W x H  
    """
    if len(img_np) == 4:
        return [img[::-1] for img in img_np]
    else:
        return img_np[::-1]


def np_to_cv2(img_np): 
    """
    Convert image in numpy.array to cv2 image.
    From C x W x H [0..1] to  W x H x C [0...255]
    """
    return np.clip(img_np.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)


def np_to_torch(img_np):
    """
    Convert image in numpy.array to torch.Tensor.
    From B x C x W x H [0..1] to  B x C x W x H [0..1]
    """
    return torch.from_numpy(np.ascontiguousarray(img_np))


def torch_to_np(img_var):
    """
    Convert an image in torch.Tensor format to numpy.array.
    From B x C x W x H [0..1] to  B x C x W x H [0..1]
    """
    return img_var.detach().cpu().numpy()


######################################################################
# Others
######################################################################

def mkdir(dir):
    os.makedirs(dir, exist_ok=True)
    os.chmod(dir, 0o777)


def build_dir(res_dir, subpath):
    mkdir(os.path.join(res_dir, subpath))
    res_type_list = ['stitch', 'full_with_boundary', 'full']
    for res_type in res_type_list:
        sub_res_dir = os.path.join(res_dir, subpath, res_type)
        mkdir(sub_res_dir)
        mkdir(os.path.join(sub_res_dir, 'sequence'))
        mkdir(os.path.join(sub_res_dir, 'batch'))


def get_model_num_parameters(model):
    """
    Return total number of parameters in model
    """
    total_num=0
    if type(model) == type(dict()):
        for key in model:
            for p in model[key].parameters():
                total_num+=p.nelement()
    else:
        for p in model.parameters():
            total_num+=p.nelement()
    return total_num