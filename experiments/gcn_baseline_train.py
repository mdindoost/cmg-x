import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
from metrics_logger import MetricsLogger
import os

# Load dataset
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

# Model
class GCNBaseline(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Training and Evaluation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNBaseline(
    in_channels=dataset.num_node_features,
    hidden_channels=64,
    out_channels=dataset.num_classes
).to(device)

data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Set up logger
os.makedirs("logs", exist_ok=True)
logger = MetricsLogger("logs/gcn_baseline_log.csv",
                       header=["epoch", "train_acc", "val_acc", "test_acc", "phi_avg", "gamma_avg", "phi_gamma_loss"])

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs

# Run
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    logger.log([epoch, train_acc, val_acc, test_acc, 0.0, 0.0, 0.0])
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")
