import torch
import torch.nn.functional as F
import os
import json
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def run_experiment(dataset_name="Pubmed", out_dir="logs/diffpool", epochs=400):
# def run_experiment(dataset_name="Citeseer", out_dir="logs/diffpool", epochs=400):
# def run_experiment(dataset_name="Cora", out_dir="logs/diffpool", epochs=400):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=f"data", name=dataset_name)
    data = dataset[0].to(device)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    class DiffPoolAutoencoder(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim, cluster_num):
            super().__init__()
            self.embed = GCNConv(in_dim, hidden_dim)
            self.assign = GCNConv(in_dim, cluster_num)
            self.decode = torch.nn.Linear(hidden_dim, in_dim)

        def forward(self, x, edge_index, batch):
            z = F.relu(self.embed(x, edge_index))
            s = F.softmax(self.assign(x, edge_index), dim=-1)
            x_dense, _ = to_dense_batch(z, batch)
            adj_dense = to_dense_adj(edge_index, batch)
            x_pooled, _, _, _ = dense_diff_pool(x_dense, adj_dense, s.unsqueeze(0))
            x_recon = torch.matmul(s, x_pooled.squeeze(0))
            return self.decode(x_recon), s

    model = DiffPoolAutoencoder(
        in_dim=dataset.num_node_features,
        hidden_dim=64,
        cluster_num=32
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses, cos_sims, snapshots = [], [], {}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        x_hat, s = model(data.x, data.edge_index, data.batch)
        loss = F.mse_loss(x_hat, data.x)

        if torch.isnan(loss):
            print(f"NaN at epoch {epoch}")
            break

        loss.backward()
        optimizer.step()

        cos_sim = F.cosine_similarity(F.normalize(x_hat, dim=1), F.normalize(data.x, dim=1)).mean().item()

        losses.append(loss.item())
        cos_sims.append(cos_sim)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, MSE: {loss.item():.6f}, CosSim: {cos_sim:.4f}")
        if epoch % 100 == 0:
            snapshots[epoch] = x_hat.detach().cpu()

    model.eval()
    x_hat, s = model(data.x, data.edge_index, data.batch)
    x_hat = x_hat.detach().cpu()
    X_orig = data.x.cpu()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(out_dir, f"{dataset_name}_diffpool_{run_id}")
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        "x_hat": x_hat,
        "x": X_orig,
        "assignment": s.detach().cpu(),
        "snapshots": snapshots
    }, os.path.join(save_dir, "reconstruction.pt"))

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump({
            "losses": losses,
            "cos_sims": cos_sims,
            "final_mse": losses[-1],
            "final_cos_sim": cos_sims[-1]
        }, f, indent=2)

    # Plotting
    plt.figure()
    plt.plot(losses)
    plt.title("DiffPool Autoencoder MSE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "mse_loss.png"))

    plt.figure()
    plt.plot(cos_sims)
    plt.title("DiffPool Autoencoder Cosine Similarity")
    plt.xlabel("Epoch")
    plt.ylabel("Mean CosSim")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "cos_sim.png"))

    node_errors = ((x_hat - X_orig) ** 2).sum(dim=1).numpy()
    plt.figure()
    plt.hist(node_errors, bins=50, alpha=0.75)
    plt.title("Per-node Reconstruction Error")
    plt.xlabel("L2 Error ||x̂ - x||²")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "node_errors_hist.png"))

    X_combined = torch.cat([X_orig, x_hat], dim=0).numpy()
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X_combined)

    plt.figure(figsize=(8, 6))
    N = len(X_orig)
    plt.scatter(X_2d[:N, 0], X_2d[:N, 1], label="Original", alpha=0.6)
    plt.scatter(X_2d[N:, 0], X_2d[N:, 1], label="Reconstructed", alpha=0.6)
    plt.title("t-SNE of Original vs. Reconstructed Features (DiffPool)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "tsne.png"))

    print(f"✅ Saved all results to {save_dir}")


if __name__ == '__main__':
    run_experiment()
