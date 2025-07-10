import numpy as np
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.linalg import svd
import csv
import argparse
import os


def build_normalized_laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    d = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(d + 1e-12))
    L = sp.diags(d) - A
    return (d_inv_sqrt @ L @ d_inv_sqrt).tocsr()


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
    _, s, _ = svd(Y, full_matrices=False)
    frob_sq = np.sum(s ** 2)
    spectral_sq = s[0] ** 2
    return frob_sq / (spectral_sq + 1e-12)


def compute_lambda_critical(k: int) -> float:
    return 2.0 / (1.0 + 0.5 * k)


def main(dataset_name: str = "Cora", d: int = 20, k_min: int = 1, k_max: int = 30):
    dataset = Planetoid(root="./data", name=dataset_name)
    data = dataset[0]
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
    L_norm = build_normalized_laplacian(A)

    np.random.seed(42)
    X = np.random.randn(data.num_nodes, d)

    log_file = f"sweep_{dataset_name.lower()}_k{str(k_min)}_{str(k_max)}.csv"
    results = []

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "lambda_critical", "energy_F2", "stable_rank"])

        for k in range(k_min, k_max + 1):
            Y = apply_spectral_filter(X, L_norm, k)
            energy = np.linalg.norm(Y, 'fro') ** 2
            rank = compute_stable_rank(Y)
            lambda_crit = compute_lambda_critical(k)
            writer.writerow([k, lambda_crit, energy, rank])
            results.append((k, lambda_crit, energy, rank))
            print(f"k={k:2d}  lambda_crit={lambda_crit:.4f}  energy={energy:.4f}  rank={rank:.2f}")

    best_by_rank = max(results, key=lambda tup: tup[3])
    print(f"\n[✓] Sweep complete. Results saved to: {log_file}")
    print(f"[Auto-Select] Best k by stable rank: k={best_by_rank[0]} (rank={best_by_rank[3]:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--kmin", type=int, default=1)
    parser.add_argument("--kmax", type=int, default=30)
    args = parser.parse_args()

    main(dataset_name=args.dataset, d=args.d, k_min=args.kmin, k_max=args.kmax)
