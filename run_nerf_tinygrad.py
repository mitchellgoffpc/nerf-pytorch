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
