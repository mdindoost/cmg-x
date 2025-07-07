import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from cmgx.core import cmgCluster
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.linalg import qr, eigh
from collections import defaultdict
import networkx as nx



def apply_spectral_filter(X: np.ndarray, L_norm: sp.spmatrix, k: int) -> np.ndarray:
    """
    Apply the spectral filter (I - 0.5M)^{k+1}x - (I - 0.5M)^k x to each column of X.
    M is the normalized Laplacian.

    Args:
        X: (n, d) matrix of random vectors
        L_norm: Normalized Laplacian (scipy sparse matrix)
        k: Filter order

    Returns:
        Y: (n, d) filtered embedding matrix
    """
    print(f"[DEBUG] Applying spectral filter with k={k}, input shape={X.shape}")
    I = sp.identity(L_norm.shape[0], format='csr')
    filter_matrix = I - 0.5 * L_norm
    Y = np.zeros_like(X)
    for j in range(X.shape[1]):
        x = X[:, j]
        power_k = x.copy()
        for _ in range(k):
            power_k = filter_matrix @ power_k
        power_k_plus_1 = filter_matrix @ power_k
        Y[:, j] = power_k_plus_1 - power_k
    print(f"[DEBUG] Spectral filtering complete. Output shape={Y.shape}")
    return Y

def reweight_graph_from_embeddings(Y: np.ndarray, edge_index: np.ndarray, threshold=0.1) -> sp.csr_matrix:
    """
    Reweight graph edges using cosine similarity of filtered embeddings.

    Args:
        Y: (n, d) node embeddings
        edge_index: (2, E) edge index array (numpy)
        threshold: similarity cutoff for retaining edges

    Returns:
        Weighted adjacency matrix in CSR format
    """
    print("[DEBUG] Reweighting graph using cosine similarity")
    sim = cosine_similarity(Y)
    rows, cols = edge_index
    weights = np.array([sim[r, c] if sim[r, c] > threshold else 0.0 for r, c in zip(rows, cols)])
    A = sp.coo_matrix((weights, (rows, cols)), shape=(Y.shape[0], Y.shape[0]))
    A_sym = A.maximum(A.T).tocsr()
    print(f"[DEBUG] Reweighted adjacency matrix has {A_sym.nnz} nonzeros")
    return A_sym

def build_normalized_laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    """
    Build normalized Laplacian M = D^{-1/2} L D^{-1/2}.

    Args:
        A: Symmetric adjacency matrix in CSR format

    Returns:
        Normalized Laplacian
    """
    print("[DEBUG] Building normalized Laplacian")
    d = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(d + 1e-8))
    L = sp.diags(d) - A
    L_norm = d_inv_sqrt @ L @ d_inv_sqrt
    print("[DEBUG] Normalized Laplacian built")
    return L_norm

def compute_restricted_eigenspace(L_norm: sp.spmatrix, Y: np.ndarray):
    """
    Project normalized Laplacian onto span(Y) and compute eigenspace.

    Args:
        L_norm: Normalized Laplacian (n x n)
        Y: Filtered vectors (n x d)

    Returns:
        eigenvectors: (n x d) approximate eigenvectors in original space
        eigenvalues: (d,) eigenvalues of projected Laplacian
        info: dict with runtime or diagnostics (placeholder)
    """
    print("[DEBUG] Computing restricted eigenspace via Rayleigh-Ritz")
    Q, _ = qr(Y, mode='economic')  # Orthonormal basis
    L_proj = Q.T @ (L_norm @ Q)
    eigenvalues, x_proj = eigh(L_proj)
    x = Q @ x_proj
    x = x / np.linalg.norm(x, axis=0, keepdims=True)
    print(f"[DEBUG] Restricted eigenspace eigenvalues: min={eigenvalues.min():.4f}, max={eigenvalues.max():.4f}")
    return x, eigenvalues, {'subspace_dim': Y.shape[1]}



def evaluate_phi_conductance(A: sp.csr_matrix, labels: np.ndarray) -> dict:
    print("[DEBUG] Evaluating φ conductance [USING CLUSTER SIZE FOR φ]")
    phi_scores = {}
    clusters = defaultdict(list)

    for i, label in enumerate(labels):
        clusters[label].append(i)

    for cid, nodes in clusters.items():
        S = set(nodes)
        cut_S = 0.0
        for u in nodes:
            neighbors = A[u].indices
            weights = A[u].data
            for v, w in zip(neighbors, weights):
                if v not in S:
                    cut_S += w
        phi = cut_S / len(S) if len(S) > 0 else 0.0
        # print("cut_S = ", cut_S)
        # print("len(S) = ", len(S))
        # print("phi = ", phi)
        phi_scores[cid] = phi
        # print(f"[DEBUG] Cluster {cid}: φ = {phi:.4f}")

    avg_phi = np.mean(list(phi_scores.values()))
    print(f"[DEBUG] Avg φ = {avg_phi:.4f}")
    return {'phi': phi_scores, 'avg_phi': avg_phi}

def cmg_filtered_clustering(data, k=10, d=20, threshold=0.1):
    """
    Full pipeline as described by Ioannis:
    - Spectral filter using polynomial smoothing
    - Cosine similarity reweighting
    - CMG clustering on reweighted graph

    Args:
        data: torch_geometric.data.Data object
        k: Filter order
        d: Embedding dimension
        threshold: Minimum similarity to keep edge

    Returns:
        cI: Cluster assignments (np.ndarray)
        nc: Number of clusters (int)
        phi_stats: φγ conductance scores
    """
    print("[DEBUG] Starting CMG filtered clustering pipeline")
    edge_index = data.edge_index.cpu().numpy()
    n = data.num_nodes

    # Step 1: Build normalized Laplacian
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=n).tocsr()
    L_norm = build_normalized_laplacian(A)

    # Step 2: Generate random vectors
    X = np.random.randn(n, d)
    print(f"[DEBUG] Generated random matrix X with shape {X.shape}")

    # Step 3: Apply spectral filter
    Y = apply_spectral_filter(X, L_norm, k)

    # Optional: compute approximate eigenspace for analysis
    # eigenvectors, eigenvalues, info = compute_restricted_eigenspace(L_norm, Y)

    # Step 4: Reweight graph
    A_reweighted = reweight_graph_from_embeddings(Y, edge_index, threshold=threshold)
    L_reweighted = sp.diags(A_reweighted.sum(axis=1).A.flatten()) - A_reweighted

    # Step 5: CMG clustering
    print("[DEBUG] Calling CMG on reweighted Laplacian")
    cI, nc = cmgCluster(L_reweighted.tocsc())
    print(f"[DEBUG] CMG clustering complete. Found {nc} clusters")

    # Step 6: φγ conductance
    phi_stats = evaluate_phi_conductance(A_reweighted, cI)
    print("[DEBUG] CMG filtered clustering pipeline complete")
    return cI, nc, phi_stats
