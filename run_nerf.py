import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import configargparse
from tqdm import tqdm, trange
from functools import partial
from load_llff import load_llff_data
from load_blender import load_blender_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255*np.clip(x,0,1)).astype(np.uint8)

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, num_freqs):
        self.out_dim = 3
        self.embed_fns = [lambda x: x]

        freq_bands = 2. ** torch.linspace(0., num_freqs-1, steps=num_freqs)
        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                self.embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                self.out_dim += 3

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = [4]

        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)] + [nn.Linear(W + (input_ch if i in self.skips else 0), W) for i in range(D-1)])
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = F.relu(self.pts_linears[i](h))
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)

        for i, l in enumerate(self.views_linears):
            h = F.relu(self.views_linears[i](h))

        rgb = self.rgb_linear(h)
        return torch.cat([rgb, alpha], -1)


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='xy')
    dirs = torch.stack([(i-K[0,2]) / K[0,0], -(j-K[1,2]) / K[1,1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return torch.stack([rays_o, rays_d], 0)

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    return torch.stack([o0,o1,o2], -1), torch.stack([d0,d1,d2], -1)

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples]).contiguous()
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    return bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

# Embed inputs and run network
def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'."""
    input_dirs = viewdirs[:,None].expand(inputs.shape)
    embedded = torch.cat([embed_fn(inputs), embeddirs_fn(input_dirs)], -1).flatten(end_dim=1)
    outputs_flat = torch.cat([fn(embedded[i:i+netchunk]) for i in range(0, embedded.shape[0], netchunk)], 0)
    return outputs_flat.view(*inputs.shape[:-1], outputs_flat.shape[-1])

# Convert network outputs to rgb
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
    """
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]

    noise = torch.randn(raw[...,3].shape) * raw_noise_std if raw_noise_std > 0. else 0.
    alpha = 1. - torch.exp(-F.relu(raw[...,3] + noise) * dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    if white_bkgd:
        acc_map = torch.sum(weights, -1)
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, weights


def render_rays(ray_batch, network_fn, network_query_fn, N_samples, perturb=0.,
                N_importance=0, network_fine=None, white_bkgd=False, raw_noise_std=0.):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified random points in time.
      N_importance: int. Number of additional times to sample along each ray. These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      rgb0: [num_rays, 3]. See rgb_map. Output for coarse model.
    """
    assert ray_batch.shape[-1] == 11
    N_rays = ray_batch.shape[0]
    rays_o, rays_d, near, far, viewdirs = ray_batch[:,0:3], ray_batch[:,3:6], ray_batch[:,6:7], ray_batch[:,7:8], ray_batch[:,8:11] # [N_rays, 3] each

    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals = near * (1.-t_vals) + far * (t_vals)
    z_vals = z_vals.expand([N_rays, N_samples])

    # get intervals between samples
    if perturb > 0.:
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        z_vals = lower + (upper - lower) * torch.rand(z_vals.shape) # stratified samples in those intervals

    # course sample
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map_0, weights = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    # fine sample
    z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.)).detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
    raw = network_query_fn(pts, viewdirs, network_fine)
    rgb_map, weights = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    return rgb_map, rgb_map_0

def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True, near=0., far=1., **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      rgb_map_0 [batch_size, 3]: See rgb_map. Output for coarse model.
    """
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, K, c2w)  # special case to render full image
    else:
        rays_o, rays_d = rays  # use provided ray batch

    # provide ray directions as input
    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)  # for forward facing scenes

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far, viewdirs], -1)

    # Render and reshape
    all_ret = [render_rays(rays[i:i+chunk], **kwargs) for i in range(0, rays.shape[0], chunk)]
    all_ret = [torch.cat([x[j] for x in all_ret], 0) for j in range(2)]  # transpose and cat
    all_ret = [torch.reshape(x, [*sh[:-1], *x.shape[1:]]) for x in all_ret]
    return all_ret


def config_parser(argv=sys.argv):
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
    parser.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')

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


def train(args):
    # Load data
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, _ = load_llff_data(args.datadir, args.factor, recenter=True, bd_factor=.75, spherify=args.spherify)
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

    # Create output directory
    logdir = os.path.join(args.basedir, args.expname)
    os.makedirs(logdir, exist_ok=True)

    # Cast intrinsics to right types
    H, W = int(H), int(W)
    K = np.array([[focal, 0,     0.5*W],
                  [0,     focal, 0.5*H],
                  [0,     0,     1]])

    # Create nerf model
    embed_fn = Embedder(args.multires)
    embeddirs_fn = Embedder(args.multires_views)
    model = NeRF(args.netdepth, args.netwidth, embed_fn.out_dim, embeddirs_fn.out_dim).to(device)
    model_fine = NeRF(args.netdepth_fine, args.netwidth_fine, embed_fn.out_dim, embeddirs_fn.out_dim).to(device)
    network_query_fn = partial(run_network, embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, netchunk=args.netchunk)
    optimizer = torch.optim.Adam(params=(*model.parameters(), *model_fine.parameters()), lr=args.lrate, betas=(0.9, 0.999))

    render_kwargs_train = {
        'network_query_fn' : network_query_fn, 'network_fn' : model, 'network_fine' : model_fine,
        'perturb': args.perturb, 'N_importance' : args.N_importance, 'N_samples' : args.N_samples,
        'white_bkgd' : args.white_bkgd, 'raw_noise_std' : args.raw_noise_std, 'ndc': args.dataset_type == 'llff',
        'near': near, 'far': far}
    render_kwargs_test = {**render_kwargs_train, 'perturb': False, 'raw_noise_std': 0.}

    # Prepare raybatch tensor
    N_rand = args.N_rand
    rays = torch.stack([get_rays(H, W, torch.as_tensor(K), torch.as_tensor(p)) for p in poses[:,:3,:4]], 0).numpy() # [N, ro+rd, H, W, 3]
    rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
    rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.astype(np.float32)
    np.random.shuffle(rays_rgb)
    i_batch = 0

    # Move data to GPU
    poses = torch.Tensor(poses).to(device)
    render_poses = torch.Tensor(render_poses).to(device)
    images = torch.Tensor(images).to(device)
    rays_rgb = torch.Tensor(rays_rgb).to(device)

    for i in trange(args.N_iters):

        # Sample random ray batch over all images
        batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_s = batch[:2], batch[2]

        i_batch += N_rand
        if i_batch >= rays_rgb.shape[0]:  # Shuffle data after each epoch
            rand_idx = torch.randperm(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx]
            i_batch = 0

        # Core optimization loop
        rgb, rgb0 = render(H, W, K, chunk=args.chunk, rays=batch_rays, **render_kwargs_train)
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        img_loss0 = img2mse(rgb0, target_s)
        loss = img_loss + img_loss0
        psnr = mse2psnr(img_loss)
        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        # Update learning rate
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (i / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        # Rest is logging
        if (i+1) % args.i_weights == 0:
            path = os.path.join(logdir, f'{i+1:06d}.ckpt')
            torch.save({
                'global_step': i,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if (i+1) % args.i_video == 0:
            rgbs = []
            for i, c2w in enumerate(tqdm(render_poses[:1])):
                with torch.no_grad():
                    rgb, _ = render(H, W, K, chunk=args.chunk, c2w=c2w[:3,:4], **render_kwargs_test)
                rgbs.append(rgb.cpu().numpy())
            rgbs = np.stack(rgbs, 0)
            np.savez_compressed(os.path.join(logdir, f'{args.expname}_spiral_{i+1:06d}_rgb.npz'), to8b(rgbs))

        if (i+1) % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i+1} Loss: {loss.item()}  PSNR: {psnr.item()}")


if __name__=='__main__':
    train(config_parser())
