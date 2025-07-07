import torch
import pytest
from cmgx.torch_interface import cmg_unpool_features

@pytest.mark.parametrize("method", ["copy", "mean", "first", "random", "central"])
def test_unpool_methods_edgecases(method):
    # Simulate a small P and cluster features
    X_c = torch.randn(3, 4)
    cI = torch.tensor([0, 0, 1, 2, 2, 2])  # 3 clusters: [0,0], [1], [2,2,2]
    P = torch.nn.functional.one_hot(cI, num_classes=3).float()

    if method in ["first", "random", "central"]:
        degree = torch.tensor([2, 1, 3, 4, 2, 5])  # arbitrary degrees
        out = cmg_unpool_features(X_c, P, method=method, cluster_assignments=cI, degree=degree if method == "central" else None)
    else:
        out = cmg_unpool_features(X_c, P, method=method)

    assert out.shape == (6, 4)
    assert not torch.isnan(out).any()
