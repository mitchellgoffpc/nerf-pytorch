import unittest
import numpy as np
import torch
from tinygrad.tensor import Tensor
from run_nerf import Embedder as TorchEmbedder, NeRF as TorchNeRF
from run_nerf_tinygrad import Embedder as TinyEmbedder, NeRF as TinyNeRF


class TestEmbedder(unittest.TestCase):
    def test_embedder(self):
        np.random.seed(42)
        torch_embed = TorchEmbedder(10)
        tiny_embed = TinyEmbedder(10)
        data = np.random.normal(0, 1, size=(32, 3)).astype(np.float32)
        torch_result = torch_embed(torch.as_tensor(data)).numpy()
        tiny_result = tiny_embed(data)
        np.testing.assert_allclose(torch_result, tiny_result, atol=1e-7)

class TestModel(unittest.TestCase):
    def test_model(self):
        np.random.seed(42)
        torch.manual_seed(42)
        torch_nerf = TorchNeRF()
        tiny_nerf = TinyNeRF()

        # Copy the weights
        for k,v in torch_nerf._modules.items():
            if isinstance(v, torch.nn.ModuleList):
                for i, (_,x) in enumerate(v._modules.items()):
                    tiny_nerf.__dict__[k][i]['weight'] = Tensor(x.weight.data.numpy().T)
                    tiny_nerf.__dict__[k][i]['bias'] = Tensor(x.bias.data.numpy().T)
            else:
                tiny_nerf.__dict__[k]['weight'] = Tensor(v.weight.data.numpy().T)
                tiny_nerf.__dict__[k]['bias'] = Tensor(v.bias.data.numpy().T)

        data = np.random.normal(0, 1, size=(32, 6)).astype(np.float32)
        torch_result = torch_nerf(torch.as_tensor(data)).detach().numpy()
        tiny_result = tiny_nerf(Tensor(data)).numpy()
        np.testing.assert_allclose(torch_result, tiny_result, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
