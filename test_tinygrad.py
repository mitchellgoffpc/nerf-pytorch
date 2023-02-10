import unittest
import numpy as np
import torch
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import SGD, get_parameters, get_named_parameters

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

        # Initialize nerfs
        torch_nerf = TorchNeRF()
        tiny_nerf = TinyNeRF()
        torch_params = dict(torch_nerf.named_parameters())
        tiny_params = get_named_parameters(tiny_nerf)
        for k in torch_params | tiny_params:
          tiny_params[k].assign(torch_params[k].detach().numpy().transpose())
        torch_optim = torch.optim.SGD(torch_nerf.parameters(), lr=0.001)
        tiny_optim = SGD(get_parameters(tiny_nerf), lr=0.001)

        # Test forward pass
        pts = np.random.normal(0, 1, size=(32, 16, 3)).astype(np.float32)
        views = np.random.normal(0, 1, size=(32, 3)).astype(np.float32)
        torch_result = torch_nerf(torch.as_tensor(pts), torch.as_tensor(views))
        tiny_result = tiny_nerf(pts, views)
        np.testing.assert_allclose(torch_result.detach().numpy(), tiny_result.numpy(), atol=1e-7)

        # Test backward pass
        torch_result.sum().backward()
        tiny_result.sum().backward()
        for k in torch_params | tiny_params:
          np.testing.assert_allclose(torch_params[k].grad.numpy().transpose(), tiny_params[k].grad.numpy(), atol=2e-4)

        # Test optimizer step
        torch_optim.step()
        tiny_optim.step()
        for k in torch_params | tiny_params:
          np.testing.assert_allclose(torch_params[k].detach().numpy().transpose(), tiny_params[k].numpy(), atol=2e-7)


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
        raw = np.random.uniform(size=(128, 64, 4))
        z_vals = np.random.uniform(size=(128, 64))
        rays_d = np.random.uniform(size=(128, 3))
        torch_raw = torch.nn.Parameter(torch.Tensor(raw), requires_grad=True)
        tiny_raw = Tensor(raw, requires_grad=True)
        torch_rgb, torch_weights = raw2outs_torch(torch_raw, torch.as_tensor(z_vals), torch.as_tensor(rays_d), white_bkgd=True)
        tiny_rgb, tiny_weights = raw2outs_tiny(tiny_raw, z_vals, rays_d, white_bkgd=True)
        torch.cat([torch_rgb, torch_weights], -1).sum().backward()
        tiny_rgb.cat(tiny_weights, dim=-1).sum().backward()
        np.testing.assert_allclose(torch_rgb.detach(), tiny_rgb.numpy(), atol=1e-5)
        np.testing.assert_allclose(torch_weights.detach(), tiny_weights.numpy(), atol=1e-5)
        np.testing.assert_allclose(torch_raw.grad.numpy(), tiny_raw.grad.numpy(), atol=3e-5)

    def test_render_rays(self):
        from run_nerf import render_rays as render_rays_torch, NeRF as TorchNeRF
        from run_nerf_tinygrad import render_rays as render_rays_tiny, NeRF as TinyNeRF

        # Initialize nerf
        torch_nerf = TorchNeRF()
        tiny_nerf = TinyNeRF()
        torch_params = dict(torch_nerf.named_parameters())
        tiny_params = get_named_parameters(tiny_nerf)
        for k in torch_params | tiny_params:
          tiny_params[k].assign(torch_params[k].detach().numpy().transpose())

        # Test forward pass
        rays = np.random.uniform(size=(128, 11)).astype(np.float32)
        torch_rgb, torch_rgb0 = render_rays_torch(torch.as_tensor(rays), torch_nerf, torch_nerf, 64, 64, False, 0.0, False)
        tiny_rgb, tiny_rgb0 = render_rays_tiny(rays, tiny_nerf, tiny_nerf, 64, 64, False, 0.0, False)
        np.testing.assert_allclose(torch_rgb0.detach(), tiny_rgb0.numpy(), atol=1e-5)
        np.testing.assert_allclose(torch_rgb.detach(), tiny_rgb.numpy(), atol=1e-5)

        # Test backward pass
        torch.cat([torch_rgb, torch_rgb0], -1).sum().backward()
        tiny_rgb.cat(tiny_rgb0, dim=-1).sum().backward()
        for k in torch_params | tiny_params:
          np.testing.assert_allclose(torch_params[k].grad.numpy().transpose(), tiny_params[k].grad.numpy(), atol=1e-4, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
