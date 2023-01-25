import unittest
import numpy as np
import torch
from tinygrad.tensor import Tensor

# Copy weights from the torch model into the tinygrad model
def copy_weights(torch_nerf, tiny_nerf):
    for k,v in torch_nerf._modules.items():
        if isinstance(v, torch.nn.ModuleList):
            for i, (_,x) in enumerate(v._modules.items()):
                tiny_nerf.__dict__[k][i]['weight'] = Tensor(x.weight.data.numpy().T)
                tiny_nerf.__dict__[k][i]['bias'] = Tensor(x.bias.data.numpy().T)
        else:
            tiny_nerf.__dict__[k]['weight'] = Tensor(v.weight.data.numpy().T)
            tiny_nerf.__dict__[k]['bias'] = Tensor(v.bias.data.numpy().T)


class TestEmbedder(unittest.TestCase):
    def test_embedder(self):
        from run_nerf import Embedder as TorchEmbedder
        from run_nerf_tinygrad import Embedder as TinyEmbedder
        np.random.seed(42)
        torch_embed = TorchEmbedder(10)
        tiny_embed = TinyEmbedder(10)
        data = np.random.normal(0, 1, size=(32, 3)).astype(np.float32)
        torch_result = torch_embed(torch.as_tensor(data)).numpy()
        tiny_result = tiny_embed(data)
        np.testing.assert_allclose(torch_result, tiny_result, atol=1e-7)

class TestModel(unittest.TestCase):
    def test_model(self):
        from run_nerf import NeRF as TorchNeRF
        from run_nerf_tinygrad import NeRF as TinyNeRF
        np.random.seed(42)
        torch.manual_seed(42)
        torch_nerf = TorchNeRF()
        tiny_nerf = TinyNeRF()
        copy_weights(torch_nerf, tiny_nerf)
        data = np.random.normal(0, 1, size=(32, 6)).astype(np.float32)
        torch_result = torch_nerf(torch.as_tensor(data)).detach().numpy()
        tiny_result = tiny_nerf(Tensor(data)).numpy()
        np.testing.assert_allclose(torch_result, tiny_result, atol=1e-7)

class TestRayHelpers(unittest.TestCase):
    def test_get_rays(self):
        from run_nerf import get_rays as get_rays_torch
        from run_nerf_tinygrad import get_rays as get_rays_tiny
        np.random.seed(42)
        H, W = 378, 504
        K = np.random.uniform(0, 1, size=(3, 3)).astype(np.float32)
        c2w = np.random.uniform(0, 1, size=(3, 4)).astype(np.float32)
        torch_result = get_rays_torch(H, W, torch.as_tensor(K), torch.as_tensor(c2w))
        tiny_result = get_rays_tiny(H, W, K, c2w)
        np.testing.assert_allclose(torch_result, tiny_result, atol=3e-4)

    def test_ndc_rays(self):
        from run_nerf import ndc_rays as ndc_rays_torch
        from run_nerf_tinygrad import ndc_rays as ndc_rays_tiny
        np.random.seed(42)
        H, W, focal, near = 378, 504, 408., 1.
        rays_o = np.random.uniform(0, 1, size=(32, 3))
        rays_d = np.random.uniform(0, 1, size=(32, 3))
        torch_rays_o, torch_rays_d = ndc_rays_torch(H, W, focal, near, torch.as_tensor(rays_o), torch.as_tensor(rays_d))
        tiny_rays_o, tiny_rays_d = ndc_rays_tiny(H, W, focal, near, rays_o, rays_d)
        np.testing.assert_allclose(torch_rays_o, tiny_rays_o, atol=1e-7)
        np.testing.assert_allclose(torch_rays_d, tiny_rays_d, atol=1e-7)

    def test_sample_pdf(self):
        from run_nerf import sample_pdf as sample_pdf_torch
        from run_nerf_tinygrad import sample_pdf as sample_pdf_tiny
        np.random.seed(42)
        N_samples = 64
        bins = np.random.uniform(0, 1, size=(32, 63))
        weights = np.random.uniform(0, 1, size=(32, 62))
        torch_result = sample_pdf_torch(torch.as_tensor(bins), torch.as_tensor(weights), N_samples, det=True)
        tiny_result = sample_pdf_tiny(bins, weights, N_samples, det=True)
        np.testing.assert_allclose(torch_result, tiny_result, atol=3e-5)

class TestRendering(unittest.TestCase):
    def test_raw2outputs(self):
        from run_nerf import raw2outputs as raw2outs_torch
        from run_nerf_tinygrad import raw2outputs as raw2outs_tiny
        raw = np.random.uniform(size=(1024, 64, 4))
        z_vals = np.random.uniform(size=(1024, 64))
        rays_d = np.random.uniform(size=(1024, 3))
        torch_rgb, torch_weights = raw2outs_torch(torch.as_tensor(raw), torch.as_tensor(z_vals), torch.as_tensor(rays_d))
        tiny_rgb, tiny_weights = raw2outs_tiny(raw, z_vals, rays_d)
        np.testing.assert_allclose(torch_rgb, tiny_rgb, atol=1e-7)
        np.testing.assert_allclose(torch_weights, tiny_weights, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
