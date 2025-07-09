'''
This test does not involve an encoder or decoder — 
it purely tests how well multilevel CMG pooling and unpooling preserves node features across coarsening levels.

It runs the following pipeline:

1. Start with node features X

2. Apply cmg_multilevel(L, levels=3) to compute a hierarchy of coarsening matrices: P1, P2, ...

3. Pool and unpool features level-by-level:

X1 = P1ᵀ X     →   X̂1 = P1 X1
X2 = P2ᵀ X1    →   X̂2 = P2 X2

4. Measure reconstruction MSE at each level:


||X̂1 - X||², ||X̂2 - X||², ...

📊 What Citeseer_multilevel_recon.png Shows
The figure plots:

X-axis: Level (e.g., 1, 2, 3)

Y-axis: Reconstruction MSE after pooling & unpooling at that level

Example:


Level 1: MSE = 2100.4
Level 2: MSE = 0.0028
Level 3: MSE = 0.0014

This often shows that:

Level 1 is very lossy (flat clusters)

Level 2-3 compress while preserving more structure

✅ Why It's Important:
This validates the CMG hierarchy:

Shows how much information is retained at coarser levels

Supports hierarchical GNN designs or multiscale routing

It's also a core theoretical test for the CMG paper — proving that the multilevel P matrices work as designed.





'''


import torch
import os
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from cmgx.pyg_pool import CMGPooling
from cmgx.pyg_unpool import CMGUnpooling
from cmgx.torch_interface import cmg_multilevel, cmg_pool_features

def run_multilevel_reconstruction_experiment(dataset_name="Citeseer", levels=3, save_dir="logs/multilevel"):
# def run_multilevel_reconstruction_experiment(dataset_name="Cora", levels=3, save_dir="logs/multilevel"):
    os.makedirs(save_dir, exist_ok=True)

    dataset = Planetoid(root=f"{dataset_name}", name=dataset_name)
    data = dataset[0]
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    # Compute Laplacian
    edge_index = data.edge_index
    N = data.num_nodes
    edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    row, col = edge_index
    deg = torch.zeros(N, device=edge_index.device).scatter_add(0, row, edge_weight)
    L = torch.sparse_coo_tensor(
        torch.stack([row, col]), -edge_weight, size=(N, N)
    )
    L = L + torch.sparse_coo_tensor(torch.stack([torch.arange(N), torch.arange(N)]), deg, size=(N, N))

    # Multilevel coarsening
    P_list = cmg_multilevel(L, levels=levels)
    recon_errors = []

    X = data.x.float()
    for level, P in enumerate(P_list):
        if isinstance(P, list):
            P = P[0]
        # Coarsen and uncoarsen
        X_coarse = cmg_pool_features(X, P, method='mean')
        X_hat = P @ X_coarse
        mse = torch.mean((X_hat - X) ** 2).item()
        recon_errors.append(mse)
        print(f"Level {level+1}: MSE = {mse:.6f}")

    # Plot
    plt.figure()
    plt.plot(range(1, len(recon_errors)+1), recon_errors, marker='o')

    plt.title(f"CMG Multilevel Reconstruction MSE ({dataset_name})")
    plt.xlabel("Level")
    plt.ylabel("MSE")
    plt.grid(True)
    out_path = os.path.join(save_dir, f"{dataset_name}_multilevel_recon.png")
    plt.savefig(out_path)
    print(f"✅ Saved multilevel plot to {out_path}")

if __name__ == "__main__":
    run_multilevel_reconstruction_experiment()
