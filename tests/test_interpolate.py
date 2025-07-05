import torch
import scipy.sparse as sp
import numpy as np
from cmgx.torch_interface import cmg_pool, cmg_interpolate, cmg_multilevel, cmg_interpolate_multilevel


def test_interpolate_single():
    A = sp.diags([1, -1], [0, -1], shape=(6, 6))
    A = (A + A.T).tocoo()
    A.setdiag(-A.sum(axis=1).A.ravel())

    indices = torch.tensor(np.array([A.row, A.col]), dtype=torch.long)
    values = torch.tensor(A.data, dtype=torch.float32)
    L = torch.sparse_coo_tensor(indices, values, size=A.shape)

    X = torch.arange(6.0).unsqueeze(1)
    Xc, Lc, P = cmg_pool(X, L)
    X_interp = cmg_interpolate(Xc, P)

    assert X_interp.shape == X.shape
    assert torch.allclose(X_interp.sum(), X.sum(), atol=1e-4)


def test_interpolate_multilevel():
    A = sp.diags([1, -1], [0, -1], shape=(8, 8))
    A = (A + A.T).tocoo()
    A.setdiag(-A.sum(axis=1).A.ravel())

    indices = torch.tensor(np.array([A.row, A.col]), dtype=torch.long)
    values = torch.tensor(A.data, dtype=torch.float32)
    L = torch.sparse_coo_tensor(indices, values, size=A.shape)

    X = torch.arange(8.0).unsqueeze(1)
    Ls, Ps = cmg_multilevel(L, levels=3)

    Xc = torch.mean(X, dim=0, keepdim=True).expand(Ls[-1].shape[0], -1)  # dummy coarse feature
    X_interp = cmg_interpolate_multilevel(Xc, Ps)

    assert X_interp.shape == X.shape

