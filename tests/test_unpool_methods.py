import torch
import numpy as np
import pytest
from cmgx.torch_interface import cmg_unpool_features

@pytest.fixture
def dummy_data():
    torch.manual_seed(0)
    N, C, F = 10, 3, 4
    cI = torch.tensor([0,0,1,1,1,2,2,2,0,1])
    P = torch.nn.functional.one_hot(cI, num_classes=C).float()
    X_c = torch.randn(C, F)
    degree = torch.tensor([2,1,3,4,2,3,1,5,0,1])
    return X_c, P, cI, degree

@pytest.mark.parametrize("method", ['copy', 'mean', 'first', 'random', 'central'])
def test_unpool_methods(dummy_data, method):
    X_c, P, cI, degree = dummy_data

    if method == 'central':
        X = cmg_unpool_features(X_c, P, method=method, cluster_assignments=cI, degree=degree)
    elif method in ['first', 'random']:
        X = cmg_unpool_features(X_c, P, method=method, cluster_assignments=cI)
    else:
        X = cmg_unpool_features(X_c, P, method=method)

    assert X.shape == (10, 4)
    assert not torch.isnan(X).any()
    assert not torch.isinf(X).any()
