import torch
from cmgx.pyg_unpool import CMGUnpooling

def test_cmg_unpooling_dense():
    # 4 nodes mapped to 2 coarse nodes
    P = torch.tensor([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ], dtype=torch.float32)

    Xc = torch.tensor([
        [10., 20.],
        [30., 40.]
    ])

    unpool = CMGUnpooling()
    X_recon = unpool(Xc, P)

    expected = torch.tensor([
        [10., 20.],
        [10., 20.],
        [30., 40.],
        [30., 40.]
    ])

    assert torch.allclose(X_recon, expected)

def test_cmg_unpooling_sparse():
    P_dense = torch.tensor([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ], dtype=torch.float32)
    P_sparse = P_dense.to_sparse()

    Xc = torch.tensor([
        [10., 20.],
        [30., 40.]
    ])

    unpool = CMGUnpooling()
    X_recon = unpool(Xc, P_sparse)

    expected = torch.tensor([
        [10., 20.],
        [10., 20.],
        [30., 40.],
        [30., 40.]
    ])

    assert torch.allclose(X_recon, expected)
