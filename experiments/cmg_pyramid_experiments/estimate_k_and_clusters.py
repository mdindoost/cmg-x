import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import svd
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix
import argparse
import torch

def build_normalized_laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    d = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(d + 1e-12))
    L = sp.csr_matrix(sp.diags(d) - A)
    L_norm = d_inv_sqrt @ L @ d_inv_sqrt
    if not sp.issparse(L_norm):
        L_norm = sp.csr_matrix(L_norm)
    return L_norm


def apply_spectral_filter(X: np.ndarray, L_norm: sp.spmatrix, k: int) -> np.ndarray:
    I = sp.identity(L_norm.shape[0], format='csr')
    filter_matrix = I - 0.5 * L_norm
    Y = np.zeros_like(X)
    for j in range(X.shape[1]):
        x = X[:, j].copy()
        for _ in range(k):
            x = filter_matrix @ x
        x_next = filter_matrix @ x
        Y[:, j] = x_next - x
    return Y


def compute_stable_rank(Y: np.ndarray) -> float:
    u, s, vt = svd(Y, full_matrices=False)
    frob_norm_sq = np.sum(s ** 2)
    spectral_norm_sq = s[0] ** 2
    return frob_norm_sq / (spectral_norm_sq + 1e-12)


def estimate_clusters_from_filtered_response(Y: np.ndarray) -> int:
    stable_rank = compute_stable_rank(Y)
    return int(np.round(stable_rank))


def compute_lambda_critical(k: int) -> float:
    return 2.0 / (1.0 + 0.5 * k)


# -------------------------------
# ✅ Reusable function for scripts
# -------------------------------
def estimate_k(adj, X=None, k=30, d=30):
    A = adj.to_scipy().tocsr() if hasattr(adj, 'to_scipy') else adj
    L_norm = build_normalized_laplacian(A)

    if X is None:
        X = np.random.randn(A.shape[0], d)
    elif isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    Y = apply_spectral_filter(X, L_norm, k)
    rank_est = estimate_clusters_from_filtered_response(Y)
    lambda_crit = compute_lambda_critical(k)
    energy = np.linalg.norm(Y, 'fro') ** 2

    return {
        "Y": Y,
        "k": rank_est,
        "lambda_crit": lambda_crit,
        "energy": energy
    }

# -------------------------------
# ✅ CLI for interactive testing
# -------------------------------
def main(dataset_name: str = "Cora", k: int = 10, d: int = 20):
    dataset = Planetoid(root="./data", name=dataset_name)
    data = dataset[0]
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()

    print(f"[INFO] Graph: {data.num_nodes} nodes, {A.nnz} edges")

    L_norm = build_normalized_laplacian(A)
    np.random.seed(42)
    X = np.random.randn(data.num_nodes, d)

    Y = apply_spectral_filter(X, L_norm, k)
    lambda_crit = compute_lambda_critical(k)
    energy = np.linalg.norm(Y, 'fro') ** 2
    rank_est = estimate_clusters_from_filtered_response(Y)

    print("\n[RESULTS]")
    print(f"Filter depth (k): {k}")
    print(f"Target eigenvalue (lambda_critical): {lambda_crit:.4f}")
    print(f"Filtered response energy ||Y||_F^2: {energy:.4f}")
    print(f"Estimated stable rank: {rank_est}")
    print(f"Estimated number of clusters: {rank_est}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--d", type=int, default=20)
    args = parser.parse_args()

    main(dataset_name=args.dataset, k=args.k, d=args.d)
