import torch
import torch.nn.functional as F
from cmgx.pyg_pool import CMGPooling
from cmgx.pyg_unpool import CMGUnpooling
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = Planetoid(root="data/Cora", name="Cora")
data = dataset[0].to(device)
data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

class CMGAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GCNConv(dataset.num_node_features, 64)
        self.pool = CMGPooling()
        self.decoder = GCNConv(64, dataset.num_node_features)
        self.unpool = CMGUnpooling()

    def forward(self, x, edge_index, batch):
        x = F.relu(self.encoder(x, edge_index))
        x_c, edge_index_c, batch_c, P, _ = self.pool(x, edge_index, batch, return_P=True)
        x_d = self.decoder(x_c, edge_index_c)
        x_rec = self.unpool(x_d, P)
        return x_rec

model = CMGAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []
cos_sims = []
snapshots = {}

for epoch in range(1, 401):
    model.train()
    optimizer.zero_grad()
    x_hat = model(data.x, data.edge_index, data.batch)
    loss = F.mse_loss(x_hat, data.x)

    if torch.isnan(loss):
        print(f"NaN detected at epoch {epoch}")
        break

    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    cos_sim = F.cosine_similarity(x_hat, data.x).mean().item()
    cos_sims.append(cos_sim)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, MSE: {loss.item():.6f}, CosSim: {cos_sim:.4f}")
    if epoch % 100 == 0:
        snapshots[epoch] = x_hat.detach().cpu()

# Final output
model.eval()
x_hat = model(data.x, data.edge_index, data.batch).detach().cpu()

# Save outputs and snapshots
torch.save({
    "x_hat": x_hat,
    "x": data.x.cpu(),
    "snapshots": snapshots
}, "reconstruction_outputs.pt")

# Plot MSE loss
plt.figure()
plt.plot(losses)
plt.title("CMG-X Pool→Unpool MSE (Cora)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig("pool_unpool_reconstruction_loss.png")

# Plot cosine similarity
plt.figure()
plt.plot(cos_sims)
plt.title("CMG-X Cosine Similarity (Cora)")
plt.xlabel("Epoch")
plt.ylabel("Mean CosSim(X̂, X)")
plt.grid(True)
plt.savefig("pool_unpool_cosine_similarity.png")

# Per-node reconstruction error
node_errors = ((x_hat - data.x.cpu()) ** 2).sum(dim=1).detach().numpy()
plt.figure()
plt.hist(node_errors, bins=50, alpha=0.75)
plt.title("Per-node Reconstruction Error")
plt.xlabel("L2 Error ||x̂_i - x_i||²")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("pool_unpool_node_errors_hist.png")

# t-SNE of original vs. reconstructed features
X_orig = data.x.cpu().numpy()
X_recon = x_hat.numpy()
X_combined = torch.cat([data.x.cpu(), x_hat], dim=0).numpy()
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X_combined)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:len(X_orig), 0], X_2d[:len(X_orig), 1], label="Original", alpha=0.6)
plt.scatter(X_2d[len(X_orig):, 0], X_2d[len(X_orig):, 1], label="Reconstructed", alpha=0.6)
plt.title("t-SNE of Original vs. Reconstructed Node Features")
plt.legend()
plt.grid(True)
plt.savefig("pool_unpool_tsne_comparison.png")

print("Saved all outputs, plots, and t-SNE visualization.")
