import torch
from torch_geometric.data import Data
from cmgx.pyg_pool import CMGPooling
from cmgx.torch_interface import cmg_interpolate_multilevel


def test_cmg_pooling_layer_basic():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ], dtype=torch.long)

    x = torch.tensor([[1.], [2.], [3.], [4.]], dtype=torch.float32)
    batch = torch.tensor([0, 0, 0, 0], dtype=torch.long)

    model = CMGPooling(return_all=False)
    x_pool, edge_index_pool, batch_pool = model(x, edge_index, batch)

    assert x_pool.shape[0] < x.shape[0]
    assert edge_index_pool.shape[0] == 2
    assert batch_pool.shape[0] == x_pool.shape[0]


def test_cmg_pooling_layer_return_all():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ], dtype=torch.long)

    x = torch.tensor([[1.], [2.], [3.], [4.]], dtype=torch.float32)
    batch = torch.tensor([0, 0, 0, 0], dtype=torch.long)

    model = CMGPooling(return_all=True)
    x_pool, edge_index_pool, batch_pool, P, L_pool = model(x, edge_index, batch)

    assert x_pool.shape[0] < x.shape[0]
    assert edge_index_pool.shape[0] == 2
    assert batch_pool.shape[0] == x_pool.shape[0]

    # Flatten if P is a list
    if isinstance(P, list):
        P_flat = torch.cat(P, dim=0)
    else:
        P_flat = P

    assert P_flat.shape[0] == x.shape[0]

    # Interpolation test
    x_interp = cmg_interpolate_multilevel(x_pool, [P_flat])
    assert x_interp.shape == x.shape

