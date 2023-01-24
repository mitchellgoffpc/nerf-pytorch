import numpy as np
from tinygrad.tensor import Tensor

def linear(in_feats, out_feats):
  return {"weight": Tensor.uniform(in_feats, out_feats), "bias": Tensor.zeros(out_feats)}

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
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = [4]

        self.pts_linears = [linear(input_ch, W)] + [linear(W + (input_ch if i in self.skips else 0), W) for i in range(D-1)]
        self.views_linears = [linear(input_ch_views + W, W//2)]
        self.feature_linear = linear(W, W)
        self.alpha_linear = linear(W, 1)
        self.rgb_linear = linear(W//2, 3)

    def __call__(self, x):
        input_pts, input_views = x[:, :self.input_ch], x[:, self.input_ch:]
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = h.linear(**self.pts_linears[i]).relu()
            if i in self.skips:
                h = input_pts.cat(h, dim=-1)

        alpha = h.linear(**self.alpha_linear)
        feature = h.linear(**self.feature_linear)
        h = feature.cat(input_views, dim=-1)

        for i, l in enumerate(self.views_linears):
            h = h.linear(**self.views_linears[i]).relu()

        rgb = h.linear(**self.rgb_linear)
        return rgb.cat(alpha, dim=-1)


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
