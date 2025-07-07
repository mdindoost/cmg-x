import torch
import scipy.sparse as sp
import numpy as np
from cmgx.core import cmgCluster
from typing import List, Tuple


def cmg_pool(X: torch.Tensor, L: torch.Tensor, batch: torch.Tensor = None):
    """
    Coarsen node features and Laplacian using CMG clustering.

    Args:
        X (Tensor): Node features [N, F] or [∑Nᵢ, F]
        L (Tensor): Sparse Laplacian [N, N] in torch.sparse_coo_tensor
        batch (Tensor): Optional [N] vector indicating graph ID per node

    Returns:
        X_coarse (Tensor): Coarsened node features [∑Ncᵢ, F]
        L_coarse (List[Tensor]): List of sparse coarsened Laplacians
        P_all (Tensor): Assignment matrix [∑Nᵢ, ∑Ncᵢ] (stacked one-hot rows)
    """
    device = X.device

    if batch is None:
        # Single graph mode
        N = X.shape[0]

        # Convert PyTorch sparse to SciPy CSC
        L_coo = L.coalesce()
        indices = L_coo.indices().cpu().numpy()
        values = L_coo.values().cpu().numpy()
        L_scipy = sp.coo_matrix((values, (indices[0], indices[1])), shape=(N, N)).tocsc()

        # Run CMG clustering
        cI, nc = cmgCluster(L_scipy)
        cI = torch.tensor(cI - 1, dtype=torch.long, device=device)

        # Feature aggregation (mean)
        X_coarse = torch.zeros(nc, X.shape[1], device=device)
        counts = torch.bincount(cI, minlength=nc).unsqueeze(1)
        X_coarse.index_add_(0, cI, X)
        X_coarse /= counts.clamp(min=1)

        # Assignment matrix
        P = torch.nn.functional.one_hot(cI, num_classes=nc).float()

        # Coarsen Laplacian
        L_dense = L.to_dense()
        L_coarse = (P.T @ L_dense @ P).to_sparse()

        return X_coarse, L_coarse, P

    # Batched mode
    all_Xc, all_Lc, all_P = [], [], []
    offset = 0
    for b in torch.unique(batch):
        idx = (batch == b).nonzero(as_tuple=True)[0]
        Xb = X[idx]
        Nb = idx.size(0)

        # Dense extraction for now (block diagonal)
        Lb = L.to_dense()[idx][:, idx]
        Lb_scipy = sp.csr_matrix(Lb.cpu().numpy()).tocsc()

        cI, nc = cmgCluster(Lb_scipy)
        cI = torch.tensor(cI - 1, dtype=torch.long, device=device)

        # Features
        Xc = torch.zeros(nc, Xb.shape[1], device=device)
        counts = torch.bincount(cI, minlength=nc).unsqueeze(1)
        Xc.index_add_(0, cI, Xb)
        Xc /= counts.clamp(min=1)

        # Assignment matrix
        Pb = torch.nn.functional.one_hot(cI, num_classes=nc).float()

        # Laplacian projection
        Lc = (Pb.T @ Lb.to(device) @ Pb).to_sparse()

        all_Xc.append(Xc)
        all_Lc.append(Lc)
        all_P.append(Pb)

        offset += Nb

    return torch.cat(all_Xc, dim=0), all_Lc, all_P

def cmg_unpool_features(X_c, P, method='copy', cluster_assignments=None, degree=None):
    """
    Unpools coarse features back to fine level using various methods.

    Args:
        X_c (Tensor): (C, F) coarse features
        P (Tensor): (N, C) assignment matrix
        method (str): 'copy', 'mean', 'central', 'random', 'first'
        cluster_assignments (Tensor): (N,) cluster index per node (required for non-matrix modes)
        degree (Tensor): (N,) node degree vector (required for 'central')

    Returns:
        X (Tensor): (N, F) fine-level reconstructed features
    """
    if isinstance(P, list):
        P = P[0]

    if method in ['copy', 'mean']:
        X = P @ X_c
        if method == 'mean':
            cluster_sizes = P.sum(dim=1).clamp(min=1).unsqueeze(1)
            X = X / cluster_sizes
        return X

    if cluster_assignments is None:
        raise ValueError("cluster_assignments must be provided for method = '{}'".format(method))

    N, F = P.shape[0], X_c.shape[1]
    X = torch.zeros((N, F), device=X_c.device)

    for cluster_id in range(X_c.shape[0]):
        members = (cluster_assignments == cluster_id).nonzero(as_tuple=False).squeeze()
        if members.numel() == 0:
            continue

        if method == 'first':
            idx = members.item() if members.ndim == 0 else members[0]

        elif method == 'random':
            idx = members.item() if members.numel() == 1 else members[torch.randint(len(members), (1,)).item()]


        elif method == 'central':
            if degree is None:
                raise ValueError("degree vector is required for 'central' method")
            if members.numel() == 1:
                idx = members.item()
            else:
                idx = members[degree[members].argmax()]

        else:
            raise ValueError(f"Unknown unpooling method: {method}")

        X[idx] = X_c[cluster_id]

    return X

    
def cmg_multilevel(L: torch.Tensor, levels: int = 3) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Build a multilevel hierarchy of coarsened Laplacians and assignment matrices.

    Args:
        L (Tensor): Input Laplacian [N, N], sparse format
        levels (int): Maximum number of coarsening levels

    Returns:
        L_hierarchy (List[Tensor]): List of Laplacians [L₀, L₁, ..., L_k]
        P_hierarchy (List[Tensor]): List of assignment matrices [P₀, ..., P_{k−1}]
            where each Pᵢ maps Lᵢ → Lᵢ₊₁
    """
    L_hierarchy = [L]
    P_hierarchy = []

    for _ in range(levels):
        L_curr = L_hierarchy[-1]
        N = L_curr.shape[0]

        # Fake features (identity) to trigger clustering
        X = torch.eye(N, dtype=torch.float32, device=L_curr.device)

        Xc, Lc, P = cmg_pool(X, L_curr)
        L_hierarchy.append(Lc)
        P_hierarchy.append(P)

        if Lc.shape[0] <= 2:
            break  # can't go further

    return L_hierarchy, P_hierarchy

def cmg_interpolate(X_coarse: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """
    Interpolate coarse features back to fine level using assignment matrix.

    Args:
        X_coarse (Tensor): [Nc, F] coarse features
        P (Tensor): [N, Nc] assignment matrix (one-hot or soft)

    Returns:
        X_fine (Tensor): [N, F] reconstructed features
    """

    return P @ X_coarse
    
def cmg_interpolate_multilevel(Xk: torch.Tensor, P_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Interpolate from coarsest level to finest using a hierarchy of assignment matrices.

    Args:
        Xk (Tensor): Feature matrix at coarsest level [Nk, F]
        P_list (List[Tensor or List[Tensor]]): Assignment matrices [P_{k−1}, ..., P₀]
            where each Pᵢ maps level i to i+1. Inner items may be lists from batch mode.

    Returns:
        X0 (Tensor): Interpolated features at finest level [N0, F]
    """

    X = Xc
    for P in reversed(P_list):
        X = P @ X
    return X
    
def cmg_interpolate_multilevel(Xk: torch.Tensor, P_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Propagate features up from coarsest level to input level using assignment matrices.

    Args:
        Xk (Tensor): Feature matrix at coarsest level [Nk, F]
        P_list (List[Tensor]): List of assignment matrices from coarse → fine

    Returns:
        X0 (Tensor): Feature matrix at input level [N0, F]
    """
    X = Xk
    for P in reversed(P_list):
        if isinstance(P, list):
            P = torch.cat(P, dim=0)

        # Handle sparse assignment matrix
        if P.is_sparse:
            X = torch.sparse.mm(P, X)
        else:
            X = P @ X

    return X

import torch

def cmg_pool_features(X, P, method='sum'):
    """
    Pools node features X using assignment matrix P.

    Args:
        X (Tensor): (N, F) fine node features
        P (Tensor): (N, C) one-hot cluster assignments
        method (str): 'sum' or 'mean'

    Returns:
        X_c (Tensor): (C, F) coarsened node features
    """
    if isinstance(P, list):  
        P = P[0]
        
    if P.dim() != 2 or X.dim() != 2:
        raise ValueError("X and P must be 2D tensors")

    if P.shape[0] != X.shape[0]:
        raise ValueError(f"P and X shape mismatch: {P.shape[0]} vs {X.shape[0]}")

    X_c = P.T @ X

    if method == 'mean':
        if P.is_sparse:
            P = P.to_dense()
        cluster_sizes = P.sum(dim=0).clamp(min=1).unsqueeze(1)

        X_c = X_c / cluster_sizes
    elif method != 'sum':
        raise ValueError(f"Invalid method: {method}. Use 'sum' or 'mean'.")

    return X_c




