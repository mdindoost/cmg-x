import torch
from cmgx.torch_interface import cmg_pool, cmg_unpool
import scipy.sparse as sp
import numpy as np


def test_cmg_unpool_reconstruction():
    # Simple path graph
    A = sp.coo_matrix([
        [ 1, -1,  0,  0],
        [-1,  2, -1,  0],
        [ 0, -1,  2, -1],
        [ 0,  0, -1,  1]
    ])
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

    indices = torch.tensor(np.array([A.row, A.col]), dtype=torch.long)
    values = torch.tensor(A.data, dtype=torch.float32)
    L = torch.sparse_coo_tensor(indices, values, size=A.shape)

    # Pool
    Xp, Lp, P = cmg_pool(X, L)

    # Unpool
    X_hat = cmg_unpool(Xp, P)

    # Since we use mean pooling, this is not identity — but should approximate
    assert X_hat.shape == X.shape
    assert torch.allclose(X_hat.sum(), X.sum(), atol=1e-5)

