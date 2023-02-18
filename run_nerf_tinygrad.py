import os
import math
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam, get_parameters
from tqdm import tqdm, trange
from load_llff import load_llff_data
from load_blender import load_blender_data
from run_nerf_helpers import load_dataset, load_args

np.random.seed(0)

# Misc
img2mse = lambda x, y: ((x - y) ** 2).mean()
mse2psnr = lambda x: -10. * x.log() / math.log(10.)
to8b = lambda x: (255*np.clip(x,0,1)).astype(np.uint8)

def linear(in_feats, out_feats):
  return {"weight": Tensor.uniform(in_feats, out_feats, requires_grad=True), "bias": Tensor.zeros(out_feats, requires_grad=True)}

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, num_freqs):
        self.out_dim = 3
        self.embed_fns = [lambda x: x]

        freq_bands = 2. ** np.linspace(0., num_freqs-1, num=num_freqs, dtype=np.float32)
        for freq in freq_bands:
            for p_fn in [np.sin, np.cos]:
                self.embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                self.out_dim += 3

    def __call__(self, inputs):
        return np.concatenate([fn(inputs) for fn in self.embed_fns], axis=-1)

# Model
class NeRF:
    def __init__(self, D=8, W=256, num_freqs=10, num_freqs_views=4):
        super().__init__()
        self.D = D
        self.W = W
        self.skips = [4]
        self.embedder = Embedder(num_freqs)
        self.embedder_views = Embedder(num_freqs_views)

        self.pts_linears = [linear(self.embedder.out_dim, W)] + [linear(W + self.embedder.out_dim if i in self.skips else W, W) for i in range(D-1)]
        self.views_linears = [linear(self.embedder_views.out_dim + W, W//2)]
        self.feature_linear = linear(W, W)
        self.alpha_linear = linear(W, 1)
        self.rgb_linear = linear(W//2, 3)

    def __call__(self, input_pts, input_views):
        embedded_pts = Tensor(self.embedder(input_pts))
        embedded_views = Tensor(self.embedder_views(input_views[:,None])).expand(*embedded_pts.shape[:-1], -1)
        embedded_pts = embedded_pts.reshape(-1, embedded_pts.shape[-1])
        embedded_views = embedded_views.reshape(-1, embedded_views.shape[-1])

        h = embedded_pts
        for i, l in enumerate(self.pts_linears):
            h = h.linear(**self.pts_linears[i]).relu()
            if i in self.skips:
                h = embedded_pts.cat(h, dim=-1)

        alpha = h.linear(**self.alpha_linear)
        feature = h.linear(**self.feature_linear)
        h = feature.cat(embedded_views, dim=-1)

        for i, l in enumerate(self.views_linears):
            h = h.linear(**self.views_linears[i]).relu()

        rgb = h.linear(**self.rgb_linear)
        rgba = rgb.cat(alpha, dim=-1)
        return rgba.reshape(*input_pts.shape[:-1], rgba.shape[-1])


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H), indexing='xy')
    dirs = np.stack([(i-K[0,2]) / K[0,0], -(j-K[1,2]) / K[1,1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], rays_d.shape)
    return np.stack([rays_o, rays_d], 0)

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

    return np.stack([o0,o1,o2], -1), np.stack([d0,d1,d2], -1)

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / np.sum(weights, axis=-1, keepdims=True)
    cdf = np.cumsum(pdf, -1)
    cdf = np.concatenate([np.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = np.linspace(0., 1., N_samples)
        u = np.broadcast_to(u, [*cdf.shape[:-1], N_samples])
        u = np.ascontiguousarray(u)
    else:
        u = np.random.uniform(size=[*cdf.shape[:-1], N_samples])

    # Invert CDF
    inds = np.stack([np.searchsorted(cdf[i], u[i], side='right') for i in range(len(bins))], axis=0)
    below = np.maximum(np.zeros_like(inds), inds-1)
    above = np.minimum(np.full_like(inds, cdf.shape[-1]-1), inds)
    inds_g = np.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = np.take_along_axis(np.broadcast_to(cdf[:,None], matched_shape), inds_g, axis=2)
    bins_g = np.take_along_axis(np.broadcast_to(bins[:,None], matched_shape), inds_g, axis=2)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = np.where(denom<1e-5, np.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    return bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])


# Convert network outputs to rgb
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = np.concatenate([dists, np.full_like(dists[...,:1], 1e10)], axis=-1)  # [N_rays, N_samples]
    dists = dists * np.linalg.norm(rays_d[...,None,:], axis=-1)
    rgb = raw[:,:,:3].sigmoid()  # [N_rays, N_samples, 3]

    noise = Tensor(np.random.normal(size=raw.shape[:-1])) * raw_noise_std if raw_noise_std > 0. else 0.
    alpha = 1. - (-(raw[:,:,3] + noise).relu() * dists)[0].exp()  # [N_rays, N_samples]  # Something is wrong with the shape here...
    alpha_comp = 1. - alpha + 1e-10
    # No cumprod :(
    alpha_cprod = alpha_comp[:,:1]
    for i in range(1, alpha.shape[-1]):
      alpha_cprod = alpha_cprod.cat(alpha_cprod[:,i-1:i] * alpha_comp[:,i:i+1], dim=-1)
    alpha_cprod = Tensor.ones(alpha.shape[0], 1).cat(alpha_cprod[:,:-1], dim=-1)
    weights = alpha * alpha_cprod
    rgb_map = (weights.reshape(*weights.shape, 1) * rgb).sum(1)  # [N_rays, 3]

    if white_bkgd:
        acc_map = weights.sum(-1).reshape(*weights.shape[:-1], 1)
        rgb_map = rgb_map + (1. - acc_map)

    return rgb_map, weights

# Volumetric rendering
def render_rays(ray_batch, network_course, network_fine, N_samples, N_importance, perturb, raw_noise_std, white_bkgd=False):
    assert ray_batch.shape[-1] == 11
    N_rays = ray_batch.shape[0]
    rays_o, rays_d, near, far, viewdirs = ray_batch[:,0:3], ray_batch[:,3:6], ray_batch[:,6:7], ray_batch[:,7:8], ray_batch[:,8:11]

    t_vals = np.linspace(0., 1., num=N_samples)
    z_vals = near * (1.-t_vals) + far * (t_vals)
    z_vals = np.broadcast_to(z_vals, [N_rays, N_samples])

    # get intervals between samples
    if perturb:
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = np.concatenate([mids, z_vals[...,-1:]], -1)
        lower = np.concatenate([z_vals[...,:1], mids], -1)
        z_vals = lower + (upper - lower) * np.random.uniform(size=z_vals.shape) # stratified samples in those intervals

    # course sample
    pts = rays_o[:,None,:] + rays_d[:,None,:] * z_vals[:,:,None] # [N_rays, N_samples, 3]
    raw = network_course(pts, viewdirs)
    rgb_map_0, weights = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    # fine sample
    z_vals_mid = .5 * (z_vals[:,1:] + z_vals[:,:-1])
    z_samples = sample_pdf(z_vals_mid, weights[:,1:-1].numpy(), N_importance, det=not perturb)
    z_vals = np.sort(np.concatenate([z_vals, z_samples], -1), -1)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
    raw = network_fine(pts, viewdirs)
    rgb_map, weights = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    return rgb_map, rgb_map_0

def render(H, W, K, rays, chunk=1024*32, ndc=True, near=0., far=1., **kwargs):
    rays_o, rays_d = rays
    viewdirs = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    viewdirs = viewdirs.reshape(-1,3)

    sh = rays_d.shape # [..., 3]
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)  # for forward facing scenes

    # Create ray batch
    rays_o = rays_o.reshape(-1,3)
    rays_d = rays_d.reshape(-1,3)
    near, far = near * np.ones_like(rays_d[...,:1]), far * np.ones_like(rays_d[...,:1])
    rays = np.concatenate([rays_o, rays_d, near, far, viewdirs], -1)

    # Render and reshape
    all_ret = [render_rays(rays[i:i+chunk], **kwargs) for i in range(0, rays.shape[0], chunk)]
    all_ret = [all_ret[0][j].cat(*[x[j] for x in all_ret[1:]], dim=0) for j in range(2)]
    all_ret = [x.reshape([*sh[:-1], *x.shape[1:]]) for x in all_ret]
    return all_ret


# Training

def train(args):
    images, poses, render_poses, H, W, focal, near, far, i_train = load_dataset(args)
    render_poses = render_poses.numpy()

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
    model_course = NeRF(args.netdepth, args.netwidth, args.multires, args.multires_views)
    model_fine = NeRF(args.netdepth_fine, args.netwidth_fine, args.multires, args.multires_views)
    optimizer = Adam([*get_parameters(model_course), *get_parameters(model_fine)], lr=args.lrate, b1=0.9, b2=0.999)

    render_kwargs_train = {
        'network_course' : model_course, 'network_fine' : model_fine,
        'N_importance' : args.N_importance, 'N_samples' : args.N_samples,
        'white_bkgd' : args.white_bkgd, 'raw_noise_std' : args.raw_noise_std, 'ndc': args.dataset_type == 'llff',
        'near': near, 'far': far, 'perturb': True}
    render_kwargs_test = {**render_kwargs_train, 'no_perturb': True, 'raw_noise_std': 0.}

    # Prepare raybatch tensor
    N_rand = args.N_rand
    rays = np.stack([get_rays(H, W, K, p) for p in poses[:,:3,:4]], axis=0) # [N, ro+rd, H, W, 3]
    rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
    rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.astype(np.float32)
    np.random.shuffle(rays_rgb)
    i_batch = 0

    for i in trange(args.N_iters):

        # Sample random ray batch over all images
        batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
        batch = np.swapaxes(batch, 0, 1)
        batch_rays, target_s = batch[:2], batch[2]

        i_batch += N_rand
        if i_batch >= rays_rgb.shape[0]:  # Shuffle data after each epoch
            rand_idx = np.random.permutation(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx]
            i_batch = 0

        # Core optimization loop
        rgb, rgb0 = render(H, W, K, batch_rays, chunk=args.chunk, **render_kwargs_train)
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
            np.savez(path, {
                'global_step': i,
                'network_fn_state_dict': render_kwargs_train['network_course'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            })
            print('Saved checkpoints at', path)

        if (i+1) % args.i_video == 0:
            rgbs = []
            for i, c2w in enumerate(tqdm(render_poses[:1])):
                rays = get_rays(H, W, K, c2w[:3,:4])
                rgb, _ = render(H, W, K, rays, chunk=args.chunk, **render_kwargs_test)
                rgbs.append(rgb.numpy())
            rgbs = np.stack(rgbs, 0)
            np.savez_compressed(os.path.join(logdir, f'{args.expname}_spiral_{i+1:06d}_rgb.npz'), to8b(rgbs))

        if (i+1) % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i+1} Loss: {loss.item()}  PSNR: {psnr.item()}")


if __name__ == '__main__':
    train(load_args())
