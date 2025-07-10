import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.utils import add_self_loops, to_networkx
from torch_geometric.nn import global_mean_pool
from metrics_logger import MetricsLogger
import os

# Load dataset
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

# Logger
os.makedirs("logs", exist_ok=True)
logger = MetricsLogger("logs/topkpool_log.csv",
                       header=["epoch", "train_acc", "val_acc", "test_acc", "phi_avg", "gamma_avg", "phi_gamma_loss"])

# Model
class TopKNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool = TopKPooling(hidden_channels, ratio=0.5)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, perm, _ = self.pool(x, edge_index, None, batch)
        x = self.conv2(x, edge_index)
        x_out = torch.zeros(batch.max().item() + 1, x.size(1), device=x.device)
        x_out[batch] = x
        return self.lin(x_out)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = TopKNet(
    in_channels=dataset.num_node_features,
    hidden_channels=64,
    out_channels=dataset.num_classes
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training
def train():
    model.train()
    optimizer.zero_grad()
    batch = torch.arange(data.num_nodes, device=device)
    out = model(data.x, data.edge_index, batch)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation
@torch.no_grad()
def test():
    model.eval()
    batch = torch.arange(data.num_nodes, device=device)
    out = model(data.x, data.edge_index, batch)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs

# Run training
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    logger.log([epoch, train_acc, val_acc, test_acc, 0.0, 0.0, 0.0])
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")
