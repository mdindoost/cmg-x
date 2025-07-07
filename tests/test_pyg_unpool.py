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


def test_cmg_unpooling_central():
    # Cluster assignments: 3 nodes → 2 clusters
    cI = torch.tensor([0, 0, 1])
    P = torch.nn.functional.one_hot(cI, num_classes=2).float()
    degree = torch.tensor([1, 5, 2])  # Node 1 should win in cluster 0

    x_c = torch.tensor([
        [10.0],  # Feature for cluster 0
        [20.0]   # Feature for cluster 1
    ])

    unpool = CMGUnpooling(method='central')
    x_rec = unpool(x_c, P, cluster_assignments=cI, degree=degree)

    expected = torch.tensor([
        [0.0],   # not selected
        [10.0],  # highest-degree node in cluster 0
        [20.0]   # only member in cluster 1
    ])

    assert torch.allclose(x_rec, expected)
