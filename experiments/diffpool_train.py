import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import DenseGCNConv, dense_diff_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj
from metrics_logger import MetricsLogger
import os

# Load PROTEINS dataset
dataset = TUDataset(root='./data', name='PROTEINS')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Logger
os.makedirs("logs", exist_ok=True)
logger = MetricsLogger("logs/diffpool_log.csv",
                       header=["epoch", "train_acc", "val_acc", "test_acc", "phi_avg", "gamma_avg", "phi_gamma_loss"])

class DiffPoolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_clusters):
        super().__init__()
        self.gnn1_pool = DenseGCNConv(in_channels, num_clusters)
        self.gnn1_embed = DenseGCNConv(in_channels, hidden_channels)
        self.gnn2_pool = DenseGCNConv(hidden_channels, num_clusters // 2)
        self.gnn2_embed = DenseGCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, adj, mask):
        s = self.gnn1_pool(x, adj)
        x = self.gnn1_embed(x, adj)
        x, adj, _, s = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x, adj, _, s = dense_diff_pool(x, adj, s)

        x = x.mean(dim=1)
        return self.lin(x)

model = DiffPoolNet(
    in_channels=dataset.num_features,
    hidden_channels=64,
    num_classes=dataset.num_classes,
    num_clusters=30
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        x, mask = to_dense_batch(batch.x, batch.batch)
        adj = to_dense_adj(batch.edge_index, batch.batch)
        out = model(x, adj, mask)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        correct += out.argmax(dim=1).eq(batch.y).sum().item()
        total += batch.y.size(0)
    return total_loss / len(loader), correct / total

@torch.no_grad()
def test():
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        batch = batch.to(device)
        x, mask = to_dense_batch(batch.x, batch.batch)
        adj = to_dense_adj(batch.edge_index, batch.batch)
        out = model(x, adj, mask)
        pred = out.argmax(dim=1)
        correct += pred.eq(batch.y).sum().item()
        total += batch.y.size(0)
    acc = correct / total
    return acc, acc, acc  # Dummy split

# Train loop
for epoch in range(1, 201):
    loss, train_acc = train()
    val_acc, test_acc, _ = test()
    logger.log([epoch, train_acc, val_acc, test_acc, 0.0, 0.0, 0.0])
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")
