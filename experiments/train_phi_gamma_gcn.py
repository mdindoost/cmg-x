import torch
import torch.nn.functional as F
import torch.nn as nn
import argparse
import os

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix

from phi_gamma_pooling import PhiGammaPooling
from metrics_logger import MetricsLogger

# --- Import spectral filter tools ---
from estimate_k_and_clusters import (
    build_normalized_laplacian,
    apply_spectral_filter,
    estimate_clusters_from_filtered_response
)

# --- Command-line arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--lambda_phi_gamma", type=float, default=0.1, help="Weight for φγ regularization")
parser.add_argument("--log_file", type=str, default="logs/cmg_phi_gamma_log.csv", help="Where to log metrics")
parser.add_argument("--k_filter", type=int, default=15, help="Spectral filter depth (for λ_critical band)")
args = parser.parse_args()
lambda_phi_gamma = args.lambda_phi_gamma
os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

# --- Load Cora dataset ---
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x, y, edge_index = data.x.to(device), data.y.to(device), data.edge_index.to(device)
adj = to_dense_adj(edge_index)[0]

# --- Estimate number of clusters using spectral filtering theory ---
print(f"[INFO] Estimating number of clusters using k={args.k_filter}")
X_rand = torch.randn((data.num_nodes, 20)).numpy()
A = to_scipy_sparse_matrix(edge_index, num_nodes=data.num_nodes).tocsr()
L = build_normalized_laplacian(A)
Y = apply_spectral_filter(X_rand, L, k=args.k_filter)
num_clusters = estimate_clusters_from_filtered_response(Y)
print(f"[AUTO] Estimated number of clusters: {num_clusters}")

# --- Define model ---
class PhiGammaGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_clusters):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool = PhiGammaPooling(hidden_channels, num_clusters=num_clusters)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, adj):
        x = F.relu(self.conv1(x, edge_index))
        x_pooled, P, phi_gamma_loss, phi_tensor, gamma_tensor = self.pool(x, adj)
        # x = F.relu(self.conv2(x_pooled, edge_index))
        x = x_pooled
        return self.lin(x), phi_gamma_loss, P, phi_tensor, gamma_tensor

model = PhiGammaGCN(
    in_channels=dataset.num_node_features,
    hidden_channels=64,
    out_channels=dataset.num_classes,
    num_clusters=num_clusters
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# --- Logger setup ---
logger = MetricsLogger(args.log_file,
                       header=["epoch", "train_acc", "val_acc", "test_acc", "phi_avg", "gamma_avg", "phi_gamma_loss"])

# --- Training loop ---
def train():
    model.train()
    optimizer.zero_grad()
    out, phi_gamma_loss, P, phi_tensor, gamma_tensor = model(x, adj)
    task_loss = F.cross_entropy(out[data.train_mask], y[data.train_mask])
    loss = task_loss + lambda_phi_gamma * phi_gamma_loss
    loss.backward()
    optimizer.step()
    return loss.item(), phi_gamma_loss.item(), phi_tensor.mean().item(), gamma_tensor.mean().item()

# --- Evaluation ---
@torch.no_grad()
def test():
    model.eval()
    out, _, _, _, _ = model(x, adj)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask].eq(y[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs  # train_acc, val_acc, test_acc

# --- Run training ---
for epoch in range(1, 201):
    loss, reg, phi_avg, gamma_avg = train()
    train_acc, val_acc, test_acc = test()
    logger.log([epoch, train_acc, val_acc, test_acc, phi_avg, gamma_avg, reg])
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Reg: {reg:.2f} | φ: {phi_avg:.4f} | γ: {gamma_avg:.4f} | Test: {test_acc:.4f}")
