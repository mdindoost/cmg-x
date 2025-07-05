import torch
from cmgx.torch_interface import cmg_pool_features, cmg_unpool_features


def test_pool_mean_unpool_mean():
    X = torch.tensor([
        [1.0],
        [3.0],
        [5.0],
        [7.0]
    ])
    P = torch.tensor([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ], dtype=torch.float32)

    X_c = cmg_pool_features(X, P, method='mean')  # should be [[2.0], [6.0]]
    X_hat = cmg_unpool_features(X_c, P, method='mean')  # broadcast cluster mean back to all nodes

    # Expected result:
    expected_X_hat = torch.tensor([
        [2.0],
        [2.0],
        [6.0],
        [6.0]
    ])
    assert torch.allclose(X_hat, expected_X_hat, atol=1e-4)
    mse = torch.nn.functional.mse_loss(X_hat, X)
    assert mse < 2.0
