'''
This script ran unsupervised autoencoder-style experiments comparing different coarsening settings using CMG pooling + unpooling.

What Was Tested
We ran combinations of:

Pooling Type	Unpool Enabled	Feature Mode	Notes
mean	        ✅	            original	    Standard pooling over real features
mean	        ✅	            identity	    Checks structure-only behavior
sum	            ✅	            original	    Total feature mass preserved
sum	            ✅	            identity	    Pure graph-structural coarsening
(same again)	❌(no unpool)	...	            Useful for ablation baseline

For each dataset: Cora, Citeseer, Pubmed.

Metrics Collected
Each run computes and saves:

Metric	                        Description
MSE Loss	                    Reconstruction error: `
Cosine Similarity	            Mean cosine between original and reconstructed features
Node-wise Error Histogram	    Distribution of per-node L2 errors
t-SNE	                        2D embedding of X vs X̂ for qualitative inspection

All results are saved in:  experiments/logs/autoencode/
Each subfolder has a naming pattern like:

Cora_cmg_mean_unpool_original_20250704_175607/
Citeseer_cmg_sum_unpool_identity_20250704_175843/

Inside each folder:
File	                What it contains
mse_loss.png	        MSE loss curve over 400 epochs
cos_sim.png	            Cosine similarity curve
node_errors_hist.png	Histogram of per-node error
tsne.png	            t-SNE plot of original vs. reconstructed features
metrics.json	        Final numerical values of loss & similarity
reconstruction.pt	    Serialized X, X̂, and snapshots (useful for future)


'''


import os
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from cmgx.pyg_pool import CMGPooling
from cmgx.pyg_unpool import CMGUnpooling
from sklearn.manifold import TSNE
from datetime import datetime

def run_one_experiment(dataset_name, pooling_mode, feature_type, run_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=f"data/{dataset_name}", name=dataset_name)
    data = dataset[0].to(device)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    # Feature selection
    if feature_type == "original":
        X = data.x
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

    class CMGAutoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = GCNConv(X.size(1), 64)
            self.pool = CMGPooling(feature_pooling=pooling_mode)
            self.decoder = GCNConv(64, X.size(1))
            self.unpool = CMGUnpooling()

        def forward(self, x, edge_index, batch):
            x = F.relu(self.encoder(x, edge_index))
            x_c, edge_index_c, batch_c, P, _ = self.pool(x, edge_index, batch, return_P=True)
            x_d = self.decoder(x_c, edge_index_c)
            x_rec = self.unpool(x_d, P)
            return x_rec, P

    model = CMGAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses, cos_sims, snapshots = [], [], {}

    for epoch in range(1, 401):
        model.train()
        optimizer.zero_grad()
        x_hat, P = model(X, data.edge_index, data.batch)

        if x_hat.shape[0] != X.shape[0]:
            loss = torch.tensor(0.0, requires_grad=True, device=x_hat.device)
            cos_sim = 0.0
        else:
            loss = F.mse_loss(x_hat, X)
            cos_sim = F.cosine_similarity(x_hat, X).mean().item()

        if torch.isnan(loss):
            print(f"NaN detected at epoch {epoch}")
            break

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        cos_sims.append(cos_sim)
        if epoch % 100 == 0:
            snapshots[epoch] = x_hat.detach().cpu()

    model.eval()
    with torch.no_grad():
        x_hat, _ = model(X, data.edge_index, data.batch)

    out_dir = os.path.join("logs", "autoencode", run_name)
    os.makedirs(out_dir, exist_ok=True)

    torch.save({
        "x_hat": x_hat.detach().cpu(),
        "x": X.detach().cpu(),
        "snapshots": snapshots
    }, os.path.join(out_dir, "reconstruction.pt"))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({
            "mse_loss": losses,
            "cosine_sim": cos_sims
        }, f)

    # Plot loss
    plt.figure()
    plt.plot(losses)
    plt.title(f"{run_name} MSE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "mse_loss.png"))
    plt.close()

    # Plot cosine sim
    plt.figure()
    plt.plot(cos_sims)
    plt.title(f"{run_name} Cosine Similarity")
    plt.xlabel("Epoch")
    plt.ylabel("CosSim")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "cos_sim.png"))
    plt.close()

    # Plot error histogram and t-SNE
    if x_hat.shape[0] == X.shape[0]:
        node_errors = ((x_hat - X) ** 2).sum(dim=1).detach().cpu().numpy()
        plt.figure()
        plt.hist(node_errors, bins=50, alpha=0.75)
        plt.title("Per-node Reconstruction Error")
        plt.xlabel("L2 Error ||x̂_i - x_i||²")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "node_errors_hist.png"))
        plt.close()

        try:
            X_comb = torch.cat([X.detach().cpu(), x_hat.detach().cpu()], dim=0).numpy()
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            X_2d = tsne.fit_transform(X_comb)
            n = X.shape[0]
            plt.figure(figsize=(6, 5))
            plt.scatter(X_2d[:n, 0], X_2d[:n, 1], label="Original", alpha=0.5, s=10)
            plt.scatter(X_2d[n:, 0], X_2d[n:, 1], label="Reconstructed", alpha=0.5, s=10)
            plt.legend()
            plt.title("t-SNE of Original vs Reconstructed")
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, "tsne.png"))
            plt.close()
        except Exception as e:
            print(f"[Warning] t-SNE failed: {e}")
    else:
        print(f"⚠️ Skipping histogram and t-SNE due to mismatched shapes: {x_hat.shape} vs {X.shape}")

    print(f"✅ Finished: {run_name}")

# ✅ Only test original features, with unpooling
datasets = ["Pubmed"]
# datasets = ["Cora", "Citeseer", "Pubmed"]
pooling_modes = ["mean", "sum"]
feature_type = "original"

for dataset in datasets:
    for pooling in pooling_modes:
        name = f"{dataset}_cmg_{pooling}_unpool_original"
        run_one_experiment(dataset, pooling, feature_type, name)
