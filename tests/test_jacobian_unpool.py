import torch
from cmgx.torch_interface import cmg_unpool

def test_cmg_unpool_jacobian_gradcheck():
    torch.manual_seed(42)
    N, F, C = 6, 3, 2  # N: fine nodes, C: coarse nodes, F: feature dim

    P = torch.tensor([
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
    ], dtype=torch.float64)

    X_coarse = torch.randn(C, F, dtype=torch.float64, requires_grad=True)

    def unpool_func(Xc):
        return cmg_unpool(Xc, P)

    assert torch.autograd.gradcheck(unpool_func, (X_coarse,), eps=1e-6, atol=1e-4), "Unpooling failed gradcheck"
