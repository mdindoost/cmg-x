import torch
import torch.nn as nn
from torch_geometric.utils import to_scipy_sparse_matrix
from cmgx.torch_interface import cmg_pool, cmg_pool_features
import scipy.sparse as sp
import numpy as np

class CMGPooling(nn.Module):
    """
    PyTorch Geometric-compatible CMG pooling layer.

    Args:
        return_all (bool): Whether to return assignment matrix and coarsened Laplacians.
        feature_pooling (str): 'sum' (default) or 'mean' — how to pool node features.
    """

    def __init__(self, return_all: bool = False, feature_pooling: str = 'sum'):
        super().__init__()
        self.return_all = return_all
        self.feature_pooling = feature_pooling

    def forward(self, x, edge_index, batch=None, return_P=False, return_all=False):
        N = x.size(0)
        device = x.device

        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)

        # Convert edge_index to Laplacian matrix
        A = to_scipy_sparse_matrix(edge_index, num_nodes=N).tocsc()
        D = A.sum(axis=1).A.ravel()
        L = sp.diags(D) - A

        row, col = L.nonzero()
        indices = torch.from_numpy(np.vstack([row, col])).long()

        values = torch.tensor(L.data, dtype=torch.float32)
        L_sparse = torch.sparse_coo_tensor(indices, values, size=(N, N)).coalesce()

        # CMG coarsening call
        x_dummy, L_pool, P_all = cmg_pool(x, L_sparse, batch)

        # Pool features with custom method
        if isinstance(P_all, list):
            x_pool = torch.cat([
                cmg_pool_features(x[batch == i], P_i, method=self.feature_pooling)
                for i, P_i in enumerate(P_all)
            ], dim=0)
        else:
            x_pool = cmg_pool_features(x, P_all, method=self.feature_pooling)

        # Build pooled edge_index and batch
        if isinstance(L_pool, list):
            edge_index_pool = []
            batch_pool = []
            offset = 0
            for i, Lc in enumerate(L_pool):
                ei = Lc.coalesce().indices() + offset
                edge_index_pool.append(ei)
                batch_pool.append(torch.full((Lc.shape[0],), i, dtype=torch.long, device=device))
                offset += Lc.shape[0]
            edge_index_pool = torch.cat(edge_index_pool, dim=1)
            batch_pool = torch.cat(batch_pool)
        else:
            edge_index_pool = L_pool.coalesce().indices()
            batch_pool = torch.zeros(x_pool.size(0), dtype=torch.long, device=device)

        # Flexible return signatures
        if return_all or return_P or self.return_all:
            return x_pool, edge_index_pool, batch_pool, P_all, L_pool
        else:
            return x_pool, edge_index_pool, batch_pool
