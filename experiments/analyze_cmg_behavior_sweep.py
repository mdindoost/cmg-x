import torch
import numpy as np
import os
import json
from datetime import datetime
from cmgx.filtered import (
    build_normalized_laplacian,
    apply_spectral_filter,
    compute_restricted_eigenspace,
    evaluate_phi_conductance,
    cmg_filtered_clustering
)
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.datasets import Planetoid

def analyze_spectrum(data, d=20, k=10):
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
    L_norm = build_normalized_laplacian(A)
    X_rand = np.random.randn(data.num_nodes, d)
    Y = apply_spectral_filter(X_rand, L_norm, k)
    x_restricted, lambda_restricted, _ = compute_restricted_eigenspace(L_norm, Y)

    try:
        from scipy.sparse.linalg import eigsh
        lambda_full, _ = eigsh(L_norm, k=min(10, data.num_nodes - 2), which='SM')
    except Exception:
        lambda_full = None

    return {
        "lambda_full": lambda_full,
        "lambda_restricted": lambda_restricted
    }

def analyze_conductance(data, cluster_labels):
    phi_stats = evaluate_phi_conductance(data, cluster_labels)
    return phi_stats

def analyze_feature_fidelity(X0: torch.Tensor, P: torch.Tensor, reduction='mean'):
    if P.is_sparse:
        P = P.to_dense()
    X1 = (P.T @ X0) / P.sum(0).clamp(min=1).unsqueeze(1)
    X0_recon = P @ X1
    mse = torch.nn.functional.mse_loss(X0_recon, X0, reduction=reduction)
    cosine_sim_all = torch.nn.functional.cosine_similarity(X0_recon, X0, dim=1)
    avg_cosine = cosine_sim_all.mean()
    node_degree = P.sum(dim=1).tolist()

    return {
        "mse": mse.item(),
        "cosine": avg_cosine.item(),
        "per_node_cosine": cosine_sim_all.tolist(),
        "node_degree": node_degree
    }

def save_result(result_dict, dataset_name, k, d, threshold, out_dir="logs/structure_eval/"):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = f"{dataset_name}_k{k}_d{d}_t{threshold}_{timestamp}"
    result_dict["config"] = {
        "dataset": dataset_name,
        "k": k,
        "d": d,
        "threshold": threshold,
        "timestamp": timestamp
    }
    fname = f"{config_str}.json"
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(serialize(result_dict), f, indent=2)
    print(f"[LOGGED] Saved {fname}")

def serialize(obj):
    if isinstance(obj, dict):
        return {str(k): serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def full_cmg_analysis(data, k=10, d=20, threshold=0.1):
    cluster_labels, nc, phi_stats, lambda_crit = cmg_filtered_clustering(
        data, k=k, d=d, threshold=threshold)
    spectrum_info = analyze_spectrum(data, d=d, k=k)
    conductance_info = phi_stats
    X0 = data.x.float()
    labels_tensor = torch.tensor(cluster_labels, dtype=torch.long)
    P = torch.nn.functional.one_hot(labels_tensor, num_classes=nc).float().to(X0.device)
    feature_info = analyze_feature_fidelity(X0, P)

    cluster_sizes = [int(c) for c in torch.bincount(labels_tensor).tolist()]
    return {
        "spectrum": spectrum_info,
        "conductance": conductance_info,
        "feature": feature_info,
        "lambda_critical": lambda_crit,
        "num_clusters": nc,
        "cluster_sizes": cluster_sizes,
        "cluster_labels": cluster_labels
    }

if __name__ == "__main__":
    datasets = ["Cora", "Citeseer", "Pubmed"]
    ks = [5, 10, 15, 20]
    d = 20
    threshold = 0.1

    for name in datasets:
        print(f"=== Dataset: {name} ===")
        ds = Planetoid(root=f"data/{name}", name=name)
        data = ds[0]
        for k in ks:
            print(f">>> Running CMG with k={k}")
            result = full_cmg_analysis(data, k=k, d=d, threshold=threshold)
            save_result(result, dataset_name=name, k=k, d=d, threshold=threshold)
