import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from estimate_k_and_clusters import estimate_k
from phi_gamma_autoencoder import PhiGammaPooling, unpool
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import ToSparseTensor, ToUndirected, Compose

# -------------------------------
# Config
# -------------------------------
use_gcn = False  # 🔁 Set to True for GCN encoder, False for MLP
dataset_name = "Cora"
hidden_dim = 64
lambda_phi_gamma = 1e-4
epochs = 200
recon_method = "soft"

# -------------------------------
# Load Data
# -------------------------------
transform = Compose([
    ToUndirected(),
    ToSparseTensor(remove_edge_index=False)
])

dataset = Planetoid(root=f'./data/{dataset_name}', name=dataset_name, transform=transform)
data = dataset[0]
x, y, adj_t, edge_index = data.x, data.y, data.adj_t, data.edge_index

# Estimate number of clusters
L = estimate_k(adj_t, x)
num_clusters = L["k"]
print(f"[INFO] Estimated number of clusters (stable rank): {num_clusters}")
print(f"[INFO] Training on {dataset_name}")

# -------------------------------
# Define Models
# -------------------------------
class PhiGammaNodeClassifierMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_clusters, num_classes, recon_method='soft'):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, hidden_channels),
            torch.nn.ReLU()
        )
        self.pool = PhiGammaPooling(hidden_channels, num_clusters)
        self.recon_method = recon_method
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, adj):
        x_enc = self.encoder(x)
        X_pooled, P, phi_gamma_loss, phi_tensor, gamma_tensor = self.pool(x_enc, adj)
        degrees = adj.sum(dim=1)
        cluster_ids = P.argmax(dim=1)
        x_unpooled = unpool(P, X_pooled, method=self.recon_method, cluster_assignments=cluster_ids, degrees=degrees)
        logits = self.classifier(x_unpooled)
        return logits, phi_gamma_loss, P, phi_tensor, gamma_tensor


class PhiGammaNodeClassifierGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_clusters, num_classes, recon_method='soft'):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, 2 * hidden_channels)
        self.gcn2 = GCNConv(2 * hidden_channels, hidden_channels)
        self.pool = PhiGammaPooling(hidden_channels, num_clusters)
        self.recon_method = recon_method
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, adj, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x_enc = F.relu(self.gcn2(x, edge_index))
        X_pooled, P, phi_gamma_loss, phi_tensor, gamma_tensor = self.pool(x_enc, adj)
        degrees = adj.sum(dim=1)
        cluster_ids = P.argmax(dim=1)
        x_unpooled = unpool(P, X_pooled, method=self.recon_method, cluster_assignments=cluster_ids, degrees=degrees)
        logits = self.classifier(x_unpooled)
        return logits, phi_gamma_loss, P, phi_tensor, gamma_tensor

# -------------------------------
# Training + Evaluation
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = (PhiGammaNodeClassifierGCN if use_gcn else PhiGammaNodeClassifierMLP)(
    in_channels=dataset.num_node_features,
    hidden_channels=hidden_dim,
    num_clusters=num_clusters,
    num_classes=dataset.num_classes,
    recon_method=recon_method
).to(device)

x, y, adj_t, edge_index = x.to(device), y.to(device), adj_t.to(device), edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def evaluate(model, split='val'):
    model.eval()
    with torch.no_grad():
        if use_gcn:
            logits, _, _, _, _ = model(x, adj_t, edge_index)
        else:
            logits, _, _, _, _ = model(x, adj_t)

        pred = logits.argmax(dim=1)
        mask = getattr(data, f'{split}_mask')
        correct = (pred[mask] == y[mask]).sum().item()
        acc = correct / mask.sum().item()
        return acc

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()

    if use_gcn:
        logits, phi_gamma_loss, P, phi_tensor, gamma_tensor = model(x, adj_t, edge_index)
    else:
        logits, phi_gamma_loss, P, phi_tensor, gamma_tensor = model(x, adj_t)

    ce_loss = F.cross_entropy(logits[data.train_mask], y[data.train_mask])
    loss = ce_loss + lambda_phi_gamma * phi_gamma_loss
    loss.backward()
    optimizer.step()

    train_acc = evaluate(model, 'train')
    val_acc = evaluate(model, 'val')
    if epoch % 10 == 0:
        print(f"[Epoch {epoch:03d}] Total Loss: {loss.item():.4f} | CE: {ce_loss.item():.4f} | ϕγ: {phi_gamma_loss.item():.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# -------------------------------
# Final Report
# -------------------------------
test_acc = evaluate(model, 'test')
print(f"\n[RESULT] Test Accuracy: {test_acc:.4f}")

print("\n========== CMG-X Run Summary ==========")
print(f"  Model:          {'GCN + PhiGammaPooling' if use_gcn else 'MLP + PhiGammaPooling'}")
print(f"  Recon method:   {recon_method}")
print(f"  Clusters used:  {num_clusters}")
print(f"  Hidden dim:     {hidden_dim}")
print(f"  λ_ϕγ_loss weight:    {lambda_phi_gamma}")
print(f"  Final CE Loss:  {ce_loss.item():.4f}")
print(f"  Final ϕγ:  {phi_gamma_loss.item():.4f}")
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Val Accuracy:   {val_acc:.4f}")
print(f"  Test Accuracy:  {test_acc:.4f}")
print("=======================================")
