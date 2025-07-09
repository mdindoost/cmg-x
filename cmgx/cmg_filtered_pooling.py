
import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from cmgx.torch_interface import cmgCluster
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import to_scipy_sparse_matrix

def build_normalized_laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    d = np.array(A.sum(axis=1)).flatten()
    d_safe = d + 1e-12
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(d_safe))
    L = sp.diags(d) - A
    L_norm = d_inv_sqrt @ L @ d_inv_sqrt
    return L_norm.tocsr()

def apply_spectral_filter(X: np.ndarray, L_norm: sp.spmatrix, k: int) -> np.ndarray:
    if not sp.isspmatrix_csr(L_norm):
        L_norm = L_norm.tocsr()
    I = sp.identity(L_norm.shape[0], format='csr')
    filter_matrix = I - 0.5 * L_norm
    Y = np.zeros_like(X)
    for j in range(X.shape[1]):
        x = X[:, j].copy()
        power_k = x.copy()
        for _ in range(k):
            power_k = filter_matrix @ power_k
        power_k_plus_1 = filter_matrix @ power_k
        Y[:, j] = power_k_plus_1 - power_k
    return Y

class CMGFilteredPooling(nn.Module):
    def __init__(self, k: int = None, filter_dim: int = 20, threshold: float = 0.1):
        super().__init__()
        self.k = k
        self.filter_dim = filter_dim
        self.threshold = threshold

    def forward(self, x, edge_index, batch=None, return_P=False, return_all=False):
        N = x.size(0)
        device = x.device
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)

        A = to_scipy_sparse_matrix(edge_index, num_nodes=N).tocsc()

        if self.k is not None:
            A_csr = A.tocsr()
            L_norm = build_normalized_laplacian(A_csr)
            X_rand = np.random.randn(N, self.filter_dim)
            Y = apply_spectral_filter(X_rand, L_norm, self.k)
            sim = cosine_similarity(Y)
            rows, cols = A.nonzero()
            weights = sim[rows, cols]
            weights = np.where(weights > self.threshold, weights, 0.0)
            A = sp.coo_matrix((weights, (rows, cols)), shape=(N, N))
            A = A.maximum(A.T).tocsc()


        degrees = np.array(A.sum(axis=1)).flatten()
        L = sp.diags(degrees) - A

        from cmgx.core import cmgCluster
        cI_raw, nc = cmgCluster(L)
        cI = torch.tensor(cI_raw - 1, dtype=torch.long, device=device)

        P = torch.nn.functional.one_hot(cI, num_classes=nc).float()

        # Reconstruct pooled edge_index
        L_dense = torch.tensor(L.toarray(), dtype=torch.float32, device=device)
        L_coarse = (P.T @ L_dense @ P).to_sparse()
        edge_index_c = L_coarse.coalesce().indices()

        if return_all or return_P:
            return x, edge_index_c, batch, P, L_coarse
        else:
            return x, edge_index_c, batch
