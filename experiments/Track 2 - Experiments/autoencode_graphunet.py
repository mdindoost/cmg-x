import os
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GraphUNet
from sklearn.manifold import TSNE
from datetime import datetime

def run_experiment(dataset_name="Cora", root_dir="data", out_dir="logs/autoencode", epochs=400):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=os.path.join(root_dir, dataset_name), name=dataset_name)
    data = dataset[0].to(device)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    class GraphUNetAutoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = GCNConv(dataset.num_node_features, 64)
            self.unet = GraphUNet(64, 64, 64, depth=1, pool_ratios=0.5)
            self.decoder = torch.nn.Linear(64, dataset.num_node_features)

        def forward(self, x, edge_index):
            x = F.relu(self.encoder(x, edge_index))
            x = self.unet(x, edge_index)
            x = self.decoder(x)
            return x

    model = GraphUNetAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses, cos_sims, snapshots = [], [], {}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        x_hat = model(data.x, data.edge_index)
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

    # Evaluation
    model.eval()
    x_hat = model(data.x, data.edge_index).detach().cpu()
    X_orig = data.x.cpu()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(out_dir, f"{dataset_name}_graphunet_{run_id}")
    os.makedirs(save_dir, exist_ok=True)

    torch.save({"x_hat": x_hat, "x": X_orig, "snapshots": snapshots}, os.path.join(save_dir, "reconstruction.pt"))

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump({"losses": losses, "cos_sims": cos_sims}, f, indent=2)

    # Plotting
    plt.figure()
    plt.plot(losses)
    plt.title("GraphUNet Autoencoder MSE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "mse_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(cos_sims)
    plt.title("GraphUNet Autoencoder Cosine Similarity")
    plt.xlabel("Epoch")
    plt.ylabel("CosSim")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "cos_sim.png"))
    plt.close()

    node_errors = ((x_hat - X_orig) ** 2).sum(dim=1).numpy()
    plt.figure()
    plt.hist(node_errors, bins=50, alpha=0.75)
    plt.title("Per-node Reconstruction Error")
    plt.xlabel("L2 Error ||x̂ - x||²")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "node_errors_hist.png"))
    plt.close()

    X_combined = torch.cat([X_orig, x_hat], dim=0).numpy()
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X_combined)
    N = len(X_orig)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:N, 0], X_2d[:N, 1], label="Original", alpha=0.6)
    plt.scatter(X_2d[N:, 0], X_2d[N:, 1], label="Reconstructed", alpha=0.6)
    plt.title("t-SNE of Original vs. Reconstructed Features (GraphUNet)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "tsne.png"))
    plt.close()

    print(f"✅ Saved all results to {save_dir}")

if __name__ == '__main__':
    run_experiment("Cora")
