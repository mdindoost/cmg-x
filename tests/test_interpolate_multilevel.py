import torch
from cmgx.torch_interface import cmg_interpolate_multilevel

def test_interpolate_multilevel():
    from cmgx.torch_interface import cmg_interpolate_multilevel

    # Coarsest-level features: 2 nodes
    Xk = torch.tensor([[10.0], [20.0]])

    # Level 1: 4 nodes → 2 coarse
    P1 = torch.tensor([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ], dtype=torch.float32)

    # Level 0: 6 nodes → 4 coarse
    P0 = torch.tensor([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    # Run interpolation
    P_list = [P0, P1]
    X_fine = cmg_interpolate_multilevel(Xk, P_list)

    assert X_fine.shape == (6, 1)
    assert torch.allclose(X_fine[:2], torch.tensor([[10.0], [10.0]]), atol=1e-4)
    assert torch.allclose(X_fine[2:4], torch.tensor([[10.0], [10.0]]), atol=1e-4)
    assert torch.allclose(X_fine[4:], torch.tensor([[20.0], [20.0]]), atol=1e-4)
