
import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import TopKPooling, SAGPooling, ASAPooling
from torch_geometric.nn import dense_diff_pool
import matplotlib.pyplot as plt


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


class GCNWithTopK(GCN):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__(in_channels, hidden_channels, out_channels)
        self.pool = TopKPooling(hidden_channels, ratio=0.8)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


class GCNWithSAGPool(GCN):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__(in_channels, hidden_channels, out_channels)
        self.pool = SAGPooling(hidden_channels, ratio=0.8)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


class GCNWithASAPPool(GCN):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__(in_channels, hidden_channels, out_channels)
        self.pool = ASAPooling(hidden_channels, ratio=0.8)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


class GCNWithDiffPool(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        from torch_geometric.nn import DenseGCNConv
        self.gnn1_pool = DenseGCNConv(in_channels, hidden_channels)
        self.gnn1_embed = DenseGCNConv(in_channels, hidden_channels)
        self.gnn2_pool = DenseGCNConv(hidden_channels, hidden_channels)
        self.gnn2_embed = DenseGCNConv(hidden_channels, hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        from torch_geometric.utils import to_dense_adj, to_dense_batch
        x, mask = to_dense_batch(data.x, data.batch)
        adj = to_dense_adj(data.edge_index, data.batch)
        s = F.relu(self.gnn1_pool(x, adj))
        x = F.relu(self.gnn1_embed(x, adj))
        x, adj, l1, e1 = dense_diff_pool(x, adj, s)
        s = F.relu(self.gnn2_pool(x, adj))
        x = F.relu(self.gnn2_embed(x, adj))
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)
        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        return self.lin2(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)

    parser.add_argument('--model', type=str, choices=['gcn', 'topk', 'diffpool', 'sagpool', 'asap'], required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    set_seed(args.seed)

    root = '/home/mohammad/cmg-x/experiments/data'
    dataset = TUDataset(root=root, name=args.dataset).shuffle()
    num_classes = dataset.num_classes
    num_features = dataset.num_features

    split = int(0.8 * len(dataset))
    train_dataset = dataset[:split]
    test_dataset = dataset[split:]
    val_split = int(0.1 * split)
    val_dataset = train_dataset[:val_split]
    train_dataset = train_dataset[val_split:]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model_map = {
        'gcn': GCN,
        'topk': GCNWithTopK,
        'sagpool': GCNWithSAGPool,
        'asap': GCNWithASAPPool,
        'diffpool': GCNWithDiffPool,
    }

    model = model_map[args.model](num_features, 64, num_classes).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logs = {'train_acc': [], 'val_acc': [], 'test_acc': []}
    best_val_acc = 0.0
    patience = 5
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(args.epochs):
        model.train()
        correct = total = 0
        for data in train_loader:
            data = data.to(args.device)
            out = model(data)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.num_graphs
        logs['train_acc'].append(correct / total)

        def eval(loader):
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for data in loader:
                    data = data.to(args.device)
                    out = model(data)
                    pred = out.argmax(dim=1)
                    correct += (pred == data.y).sum().item()
                    total += data.num_graphs
            return correct / total

        val_acc = eval(val_loader)
        test_acc = eval(test_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        logs['val_acc'].append(val_acc)
        logs['test_acc'].append(test_acc)
        print(f"Epoch {epoch:03d} | Train: {logs['train_acc'][-1]:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    log_dir = '/home/mohammad/cmg-x/experiments/logs/gnn'
    os.makedirs(log_dir, exist_ok=True)
    log_base = os.path.join(log_dir, f"{args.dataset}_{args.model}_{args.seed}")
    with open(log_base + '.json', 'w') as f:
        json.dump(logs, f, indent=2)

    plt.figure()
    plt.plot(logs['train_acc'], label='Train Acc')
    plt.plot(logs['val_acc'], label='Val Acc')
    plt.plot(logs['test_acc'], label='Test Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(log_base + '_accuracy.png')
