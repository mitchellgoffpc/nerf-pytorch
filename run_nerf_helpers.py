import sys
import numpy as np
import configargparse
from load_llff import load_llff_data
from load_blender import load_blender_data


def load_dataset(args):
    if args.dataset_type == 'llff':
        images, poses, _, render_poses, _ = load_llff_data(args.datadir, args.factor, recenter=True, bd_factor=.75, spherify=args.spherify)
        H, W, focal = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        i_test = np.arange(images.shape[0])[::args.testskip]
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if i not in i_test])
        near, far = 0., 1.

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        i_train, _, _ = i_split
        H, W, focal = hwf
        near, far = 2., 6.
        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    else:
        raise ValueError(f'Unknown dataset type {args.dataset_type}, exiting')

    return images, poses, render_poses, H, W, focal, near, far, i_train


def load_args(argv=sys.argv):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, help='number of pts sent through network in parallel, decrease if running out of memory')

    # rendering options
    parser.add_argument("--N_iters", type=int, default=200000, help='number of iterations to train')
    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=64, help='number of additional fine samples per ray')
    parser.add_argument("--multires", type=int, default=10, help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', help='options: llff / blender')
    parser.add_argument("--testskip", type=int, default=8, help='will load 1/N images from test/val sets')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, help='downsample factor for LLFF images')
    parser.add_argument("--spherify", action='store_true', help='set for spherical 360 scenes')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, help='frequency of console printout and metric logging')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')
    parser.add_argument("--i_video",   type=int, default=50000, help='frequency of render_poses video saving')

    return parser.parse_args(argv)
