import argparse
import sys
sys.path.append('src')

from inpainting_test import InpaintingTest
from configs.DIP import cfg as DIP_cfg
from configs.DIP_Vid import cfg as DIP_Vid_cfg
from configs.DIP_Vid_3DCN import cfg as DIP_Vid_3DCN_cfg
from configs.DIP_Vid_Flow import cfg as DIP_Vid_Flow_cfg


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_mode', type=str, default='DIP_Vid_Flow', help='mode of the experiment: (DIP|DIP_Vid|DIP_Vid_3DCN|DIP_Vid_Flow)', metavar='')
    parser.add_argument('--resize', nargs='+', type=int, default=None, help='height and width of the output', metavar='')
    parser.add_argument('--video_path', type=str, default='data/bmx-trees.avi', help='path of the input video', metavar='')
    parser.add_argument('--mask_path', type=str, default='data/bmx-trees_mask.avi', help='path of the input mask', metavar='')
    parser.add_argument('--res_dir', type=str, default='result', help='path to save the result', metavar='')
    parser.add_argument('--frame_sum', type=int, default=100, help='number of frames to load', metavar='')
    parser.add_argument('--dilation_iter', type=int, default=0, help='number of steps to dilate the mask', metavar='')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def main(args):
    if args.train_mode == 'DIP':
        cfg = DIP_cfg
    elif args.train_mode == 'DIP-Vid':
        cfg = DIP_Vid_cfg
    elif args.train_mode == 'DIP-Vid-3DCN':
        cfg = DIP_Vid_3DCN_cfg
    elif args.train_mode == 'DIP-Vid-Flow':
        cfg = DIP_Vid_Flow_cfg
    else:
        raise Exception("Train mode {} not implemented!".format(args.train_mode))

    cfg['resize'] = tuple(args.resize)
    if len(cfg['resize']) != 2: raise Exception("Resize must be a tuple of length 2!")
    cfg['video_path'] = args.video_path
    cfg['mask_path'] = args.mask_path
    cfg['res_dir'] = args.res_dir
    cfg['frame_sum'] = args.frame_sum
    cfg['dilation_iter'] = args.dilation_iter

    test = InpaintingTest(cfg)
    test.create_data_loader()
    test.visualize_single_batch()
    test.create_model()
    test.create_optimizer()
    test.create_loss_function()
    test.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)