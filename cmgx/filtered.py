import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from cmgx.core import cmgCluster
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.linalg import qr, eigh
from collections import defaultdict
import networkx as nx

def build_normalized_laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    """
    Build normalized Laplacian M = D^{-1/2} L D^{-1/2}.
    """
    print("[DEBUG] Building normalized Laplacian")
    d = np.array(A.sum(axis=1)).flatten()
    d_safe = d + 1e-12  # Avoid division by zero
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(d_safe))
    L = sp.diags(d) - A
    L_norm = d_inv_sqrt @ L @ d_inv_sqrt
    print("[DEBUG] Normalized Laplacian built")
    return L_norm.tocsr()

def apply_spectral_filter(X: np.ndarray, L_norm: sp.spmatrix, k: int) -> np.ndarray:
    """
    Apply the spectral filter (I - 0.5M)^{k+1}x - (I - 0.5M)^k x to each column of X.
    """
    print(f"[DEBUG] Applying spectral filter with k={k}, input shape={X.shape}")
    
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
    
    print(f"[DEBUG] Spectral filtering complete. Output shape={Y.shape}")
    return Y

def reweight_graph_from_embeddings(Y: np.ndarray, edge_index: np.ndarray, threshold=0.1) -> sp.csr_matrix:
    """
    Reweight graph edges using cosine similarity of filtered embeddings.
    """
    print("[DEBUG] Reweighting graph using cosine similarity")
    sim = cosine_similarity(Y)
    
    if edge_index.ndim == 1:
        edge_index = edge_index.reshape(2, -1)
    
    rows, cols = edge_index[0], edge_index[1]
    similarities = sim[rows, cols]
    weights = np.where(similarities > threshold, similarities, 0.0)
    
    n = Y.shape[0]
    A = sp.coo_matrix((weights, (rows, cols)), shape=(n, n))
    A_sym = A.maximum(A.T).tocsr()
    
    print(f"[DEBUG] Reweighted adjacency matrix has {A_sym.nnz} nonzeros")
    return A_sym

# Add this to your filtered.py file - replace the existing evaluate_phi_conductance function

def evaluate_phi_conductance(data_or_matrix, labels: np.ndarray, method: str = 'unweighted') -> dict:
    """
    Evaluate conductance - supports both weighted and unweighted calculations.
    
    Args:
        data_or_matrix: Either PyG data object (for unweighted) or sparse matrix (for weighted)
        labels: Cluster assignments (handles both 0-indexed and 1-indexed)
        method: 'unweighted' (default), 'weighted', or 'both'
        
    Returns:
        Dict with conductance results
    """
    
    # Handle CMG 1-indexed to 0-indexed conversion
    labels = np.array(labels)
    if labels.min() > 0:
        print(f"[DEBUG] Converting labels from 1-indexed to 0-indexed")
        labels = labels - 1
    
    if method in ['unweighted', 'both']:
        # Unweighted structural conductance (what you want)
        if hasattr(data_or_matrix, 'edge_index'):
            # PyG data object
            data = data_or_matrix
            edge_index = data.edge_index.cpu().numpy()
            num_nodes = data.num_nodes
            
            unweighted_result = _evaluate_unweighted_conductance(edge_index, num_nodes, labels)
        else:
            raise ValueError("For unweighted conductance, please provide PyG data object with edge_index")
    
    if method in ['weighted', 'both']:
        # Weighted conductance (original implementation)
        if hasattr(data_or_matrix, 'edge_index'):
            # Convert PyG data to weighted matrix
            A_weighted = to_scipy_sparse_matrix(data_or_matrix.edge_index, num_nodes=data_or_matrix.num_nodes)
            if hasattr(data_or_matrix, 'edge_weight') and data_or_matrix.edge_weight is not None:
                rows, cols = data_or_matrix.edge_index[0].numpy(), data_or_matrix.edge_index[1].numpy()
                weights = data_or_matrix.edge_weight.numpy()
                A_weighted = sp.coo_matrix((weights, (rows, cols)), shape=(data_or_matrix.num_nodes, data_or_matrix.num_nodes))
                A_weighted = A_weighted.tocsr()
        else:
            # Direct sparse matrix
            A_weighted = data_or_matrix
            if not sp.isspmatrix_csr(A_weighted):
                A_weighted = A_weighted.tocsr()
        
        weighted_result = _evaluate_weighted_conductance(A_weighted, labels)
    
    # Return based on method
    if method == 'unweighted':
        return unweighted_result
    elif method == 'weighted':
        return weighted_result
    else:  # both
        return {
            'unweighted': unweighted_result,
            'weighted': weighted_result,
            # For backward compatibility, use unweighted as primary
            **unweighted_result,
            'method': 'unweighted_primary'
        }

def _evaluate_unweighted_conductance(edge_index: np.ndarray, num_nodes: int, labels: np.ndarray) -> dict:
    """Helper: Calculate unweighted structural conductance."""
    print("[DEBUG] Computing UNWEIGHTED STRUCTURAL conductance (pure topology)")
    
    # Build unweighted adjacency matrix (all edges have weight 1.0)
    if edge_index.ndim == 1:
        edge_index = edge_index.reshape(2, -1)
    
    rows, cols = edge_index[0], edge_index[1]
    weights = np.ones(len(rows))  # All edges get weight 1.0
    
    A_unweighted = sp.coo_matrix((weights, (rows, cols)), shape=(num_nodes, num_nodes))
    A_unweighted = A_unweighted.maximum(A_unweighted.T).tocsr()
    
    print(f"[DEBUG] Unweighted graph: {num_nodes} nodes, {A_unweighted.nnz} edges")
    
    # Group nodes by cluster
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(i)
    
    total_degree = A_unweighted.sum()
    
    # Calculate conductance for each cluster
    standard_phi = {}
    normalized_phi = {}
    
    for cid, nodes in sorted(clusters.items()):
        S = set(nodes)
        complement = set(range(num_nodes)) - S
        
        print(f"\n[DEBUG] Cluster {cid}: {len(S)} nodes {sorted(list(S))}")
        
        if len(S) == 0 or len(complement) == 0:
            standard_phi[cid] = float('inf')
            normalized_phi[cid] = float('inf')
            continue
        
        # Calculate cut and degree
        cut_S = 0  # Number of edges crossing the cut
        degree_S = 0  # Total degree of nodes in S
        
        for u in S:
            row_start = A_unweighted.indptr[u]
            row_end = A_unweighted.indptr[u + 1]
            neighbors = A_unweighted.indices[row_start:row_end]
            
            degree_S += len(neighbors)
            
            # Count cut edges
            for neighbor in neighbors:
                if neighbor not in S:
                    cut_S += 1
        
        degree_complement = total_degree - degree_S
        
        # Standard conductance: φ(S) = cut(S) / min(degree(S), degree(V\S))
        min_degree = min(degree_S, degree_complement)
        standard_phi[cid] = cut_S / min_degree if min_degree > 0 else float('inf')
        
        # Normalized conductance: φ_norm(S) = cut(S) / |S|
        normalized_phi[cid] = cut_S / len(S)
        
        print(f"[DEBUG] Cluster {cid}: cut_edges={cut_S}, degree={degree_S}, "
              f"φ_std={standard_phi[cid]:.4f}, φ_norm={normalized_phi[cid]:.4f}")
    
    # Calculate averages
    finite_std = [p for p in standard_phi.values() if p != float('inf')]
    finite_norm = [p for p in normalized_phi.values() if p != float('inf')]
    
    avg_std = np.mean(finite_std) if finite_std else float('inf')
    avg_norm = np.mean(finite_norm) if finite_norm else float('inf')
    
    print(f"[DEBUG] Average unweighted conductance - Standard: {avg_std:.4f}, Normalized: {avg_norm:.4f}")
    
    return {
        'phi': standard_phi,
        'avg_phi': avg_std,
        'method': 'unweighted_standard',
        'normalized_phi': normalized_phi,
        'normalized_avg_phi': avg_norm,
        'comparison': {
            'standard_avg': avg_std,
            'normalized_avg': avg_norm,
            'ratio': avg_norm / avg_std if avg_std > 0 else float('inf')
        }
    }

def _evaluate_weighted_conductance(A_original: sp.csr_matrix, labels: np.ndarray) -> dict:
    """Helper: Calculate weighted conductance (original implementation)."""
    print(f"[DEBUG] Computing WEIGHTED conductance")
    
    # Group nodes by cluster
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(i)
    
    n = A_original.shape[0]
    total_volume = A_original.sum()
    
    # Calculate conductance for each cluster
    standard_phi = {}
    normalized_phi = {}
    
    for cid, nodes in sorted(clusters.items()):
        S = set(nodes)
        complement = set(range(n)) - S
        
        if len(S) == 0 or len(complement) == 0:
            standard_phi[cid] = float('inf')
            normalized_phi[cid] = float('inf')
            continue
        
        # Calculate cut and volume
        cut_S = 0.0
        vol_S = 0.0
        
        for u in S:
            row_start = A_original.indptr[u]
            row_end = A_original.indptr[u + 1]
            neighbors = A_original.indices[row_start:row_end]
            weights = A_original.data[row_start:row_end]
            
            # Add to volume
            vol_S += np.sum(weights)
            
            # Count cut edges
            for neighbor, weight in zip(neighbors, weights):
                if neighbor not in S:
                    cut_S += weight
        
        # Standard conductance: φ(S) = cut(S) / min(vol(S), vol(V\S))
        vol_complement = total_volume - vol_S
        min_vol = min(vol_S, vol_complement)
        standard_phi[cid] = cut_S / min_vol if min_vol > 0 else float('inf')
        
        # Normalized conductance: φ_norm(S) = cut(S) / |S|
        normalized_phi[cid] = cut_S / len(S)
    
    # Calculate averages
    finite_std = [p for p in standard_phi.values() if p != float('inf')]
    finite_norm = [p for p in normalized_phi.values() if p != float('inf')]
    
    avg_std = np.mean(finite_std) if finite_std else float('inf')
    avg_norm = np.mean(finite_norm) if finite_norm else float('inf')
    
    return {
        'phi': standard_phi,
        'avg_phi': avg_std,
        'method': 'weighted_standard',
        'normalized_phi': normalized_phi,
        'normalized_avg_phi': avg_norm,
        'comparison': {
            'standard_avg': avg_std,
            'normalized_avg': avg_norm,
            'ratio': avg_norm / avg_std if avg_std > 0 else float('inf')
        }
    }
def compute_lambda_critical(k: int) -> float:
    """Compute critical eigenvalue where filter peaks."""
    return 2.0 / (1.0 + 0.5 * k)

def compute_k_for_lambda(lambda_target: float) -> float:
    """Compute required k to target specific eigenvalue."""
    return 2.0 * (2.0 - lambda_target) / lambda_target

def compute_restricted_eigenspace(L_norm: sp.spmatrix, Y: np.ndarray):
    """
    Project normalized Laplacian onto span(Y) and compute eigenspace.
    """
    print("[DEBUG] Computing restricted eigenspace via Rayleigh-Ritz")
    Q, R = qr(Y, mode='economic')
    L_proj = Q.T @ (L_norm @ Q)
    eigenvalues, x_proj = eigh(L_proj)
    x = Q @ x_proj
    x = x / np.linalg.norm(x, axis=0, keepdims=True)
    
    print(f"[DEBUG] Eigenvalue range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
    return x, eigenvalues, {'subspace_dim': Y.shape[1]}

def cmg_filtered_clustering(data, k=10, d=20, threshold=0.1, conductance_method='both'):
    """
    Full pipeline: spectral filtering + CMG clustering + conductance evaluation.
    
    Args:
        data: torch_geometric.data.Data object
        k: Filter order
        d: Embedding dimension
        threshold: Cosine similarity threshold
        conductance_method: 'standard', 'normalized', or 'both'
    """
    print("[DEBUG] Starting CMG filtered clustering pipeline")
    print(f"[DEBUG] Parameters: k={k}, d={d}, threshold={threshold}")
    
    edge_index = data.edge_index.cpu().numpy()
    n = data.num_nodes
    
    lambda_crit = compute_lambda_critical(k)
    print(f"[DEBUG] λ_critical ≈ {lambda_crit:.4f} for filter order k = {k}")
    
    # Build adjacency matrix (NOT Laplacian)
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=n).tocsr()
    print(f"[DEBUG] Original graph: {n} nodes, {A.nnz} edges")
    
    # Build normalized Laplacian for filtering
    L_norm = build_normalized_laplacian(A)
    
    # Generate random vectors and filter
    np.random.seed(42)
    X = np.random.randn(n, d)
    Y = apply_spectral_filter(X, L_norm, k)
    
    # Reweight graph based on filtered embeddings
    A_reweighted = reweight_graph_from_embeddings(Y, edge_index, threshold=threshold)
    
    # Build Laplacian for CMG
    degrees = np.array(A_reweighted.sum(axis=1)).flatten()
    L_reweighted = sp.diags(degrees) - A_reweighted
    
    print(f"[DEBUG] Reweighted graph: {A_reweighted.nnz} edges")
    
    # Run CMG clustering
    print("[DEBUG] Calling CMG on reweighted Laplacian")
    try:
        cI_raw, nc = cmgCluster(L_reweighted.tocsc())
        cI = cI_raw - 1  # Convert from 1-indexed to 0-indexed
        print(f"[DEBUG] CMG found {nc} clusters")
        print(f"[DEBUG] Raw CMG output (1-indexed): {cI_raw}")
        print(f"[DEBUG] Converted clusters (0-indexed): {cI}")
    except Exception as e:
        print(f"[ERROR] CMG failed: {e}")
        cI = np.zeros(n, dtype=int)
        nc = 1
    
    # Evaluate conductance on reweighted graph
    #phi_stats = evaluate_phi_conductance(A_reweighted, cI, method=conductance_method)
    phi_stats = evaluate_phi_conductance(data, cI, method='unweighted')  # Use original data, not reweighted

    print("[DEBUG] CMG filtered clustering pipeline complete")
    return cI, nc, phi_stats, lambda_crit

# For backward compatibility with your existing code
def evaluate_phi_conductance_legacy(A_original: sp.csr_matrix, labels: np.ndarray) -> dict:
    """Legacy function - uses normalized conductance by default."""
    return evaluate_phi_conductance(A_original, labels, method='normalized')