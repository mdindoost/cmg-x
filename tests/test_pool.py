import torch
import scipy.sparse as sp
import numpy as np
from cmgx.torch_interface import cmg_pool


def test_cmg_pool():
    A = sp.coo_matrix([
        [ 1, -1,  0,  0],
        [-1,  2, -1,  0],
        [ 0, -1,  2, -1],
        [ 0,  0, -1,  1]
    ])
    X = torch.tensor([[1.], [2.], [3.], [4.]])

    indices = torch.tensor(np.array([A.row, A.col]), dtype=torch.long)
    values = torch.tensor(A.data, dtype=torch.float32)
    L = torch.sparse_coo_tensor(indices, values, size=A.shape)

    Xp, Lp, P = cmg_pool(X, L)

    assert Xp.shape[0] < X.shape[0]
    assert torch.is_tensor(Xp)
    assert torch.is_tensor(P)
    assert P.shape[0] == X.shape[0]
    assert torch.allclose(P.sum(dim=1), torch.ones_like(P.sum(dim=1)))  # one-hot

    if isinstance(Lp, list):
        assert all(torch.is_tensor(l) for l in Lp)
    else:
        assert torch.is_tensor(Lp)


def test_cmg_pool_batched():
    A = sp.coo_matrix([
        [ 1, -1,  0,  0],
        [-1,  2, -1,  0],
        [ 0, -1,  2, -1],
        [ 0,  0, -1,  1]
    ])
    A2 = A.copy()
    A_big = sp.block_diag([A, A2])

    X = torch.tensor([[i] for i in range(8)], dtype=torch.float32)
    batch = torch.tensor([0]*4 + [1]*4)

    indices = torch.tensor(np.array([A_big.row, A_big.col]), dtype=torch.long)
    values = torch.tensor(A_big.data, dtype=torch.float32)
    L = torch.sparse_coo_tensor(indices, values, size=A_big.shape)

    Xp, Lp, P_list = cmg_pool(X, L, batch=batch)

    assert Xp.shape[0] < X.shape[0]
    assert isinstance(Lp, list) and len(Lp) == 2
    assert isinstance(P_list, list)
    assert sum(p.shape[0] for p in P_list) == X.shape[0]
    assert all(torch.allclose(p.sum(dim=1), torch.ones_like(p.sum(dim=1))) for p in P_list)

