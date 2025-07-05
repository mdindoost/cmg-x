import torch
import scipy.sparse as sp
import numpy as np
from cmgx.torch_interface import cmg_multilevel


def test_cmg_multilevel_depth():
    # 8-node path graph
    A = sp.diags([1, -1], [0, -1], shape=(8, 8))
    A = (A + A.T).tocoo()  # ensure COO before accessing .row/.col
    A.setdiag(-A.sum(axis=1).A.ravel())
    A = A.tocoo()  # ensure it's still in COO after setdiag


    indices = torch.tensor(np.array([A.row, A.col]), dtype=torch.long)
    values = torch.tensor(A.data, dtype=torch.float32)
    L = torch.sparse_coo_tensor(indices, values, size=A.shape)

    # Build 3-level hierarchy
    Ls, Ps = cmg_multilevel(L, levels=3)

    assert len(Ls) >= 2
    assert all(isinstance(Li, torch.Tensor) for Li in Ls)
    assert all(isinstance(Pi, torch.Tensor) for Pi in Ps)
    assert all(Pi.shape[1] == Ls[i+1].shape[0] for i, Pi in enumerate(Ps))

    # Print some shape info for debug
    print("Hierarchy shapes:")
    for i, (L, P) in enumerate(zip(Ls, Ps)):
        print(f"  Level {i}: L = {L.shape}, P = {P.shape}")

