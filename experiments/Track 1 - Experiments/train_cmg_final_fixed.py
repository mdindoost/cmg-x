
import os
import argparse
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from model import GNNWithCMG
from utils import compute_metrics, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    set_seed(args.seed)
    root = '/home/mohammad/cmg-x/experiments/data'

    dataset = TUDataset(root=root, name=args.dataset).shuffle()

split = int(0.8 * len(dataset))
train_dataset = dataset[:split]
test_dataset = dataset[split:]
val_split = int(0.1 * split)
val_dataset = train_dataset[:val_split]
train_dataset = train_dataset[val_split:]

# ✅ Patch dummy node features
all_graphs = train_dataset + val_dataset + test_dataset
for data in all_graphs:
    if data.x is None:
        data.x = torch.ones((data.num_nodes, 1))

num_classes = dataset.num_classes
num_features = all_graphs[0].x.size(1)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = GNNWithCMG(num_features, 64, num_classes).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logs = {
        'train_acc': [], 'val_acc': [], 'test_acc': [],
        'compression': [], 'cos_sim': [], 'mse': [], 'cluster_stats': []
    }
    best_val_acc = 0
    best_epoch = 0
    patience = 5
    wait = 0

    for epoch in range(args.epochs):
        model.train()
        correct = total = 0
        for data in train_loader:
            data = data.to(args.device)
            out, _, _ = model(data)
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
            cos_sim_acc, mse_acc, comp_acc = 0, 0, 0
            cluster_sizes = []
            with torch.no_grad():
                for data in loader:
                    data = data.to(args.device)
                    out, P_list, x_coarse = model(data)
                    pred = out.argmax(dim=1)
                    correct += (pred == data.y).sum().item()
                    total += data.num_graphs
                    if P_list is not None:
                        cos_sim, mse, compression, sizes = compute_metrics(data, P_list, x_coarse)
                        cos_sim_acc += cos_sim * data.num_graphs
                        mse_acc += mse * data.num_graphs
                        comp_acc += compression * data.num_graphs
                        cluster_sizes.extend(sizes)
            acc = correct / total
            stats = {
                'mean': float(np.mean(cluster_sizes)) if cluster_sizes else 0,
                'std': float(np.std(cluster_sizes)) if cluster_sizes else 0,
                'min': int(np.min(cluster_sizes)) if cluster_sizes else 0,
                'max': int(np.max(cluster_sizes)) if cluster_sizes else 0,
            }
            return acc, cos_sim_acc / total, mse_acc / total, comp_acc / total, stats

        val_acc, cos_sim, mse, compression, clust_stats = eval(val_loader)
        test_acc, *_ = eval(test_loader)
        logs['val_acc'].append(val_acc)
        logs['test_acc'].append(test_acc)
        logs['cos_sim'].append(cos_sim)
        logs['mse'].append(mse)
        logs['compression'].append(compression)
        logs['cluster_stats'].append(clust_stats)

        print(f"Epoch {epoch:03d} | Train: {logs['train_acc'][-1]:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f} | CosSim: {cos_sim:.4f} | MSE: {mse:.6f} | Compression: {compression:.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹️ Early stopping at epoch {epoch} (best val @ {best_epoch})")
                break

    model.load_state_dict(best_model_state)

    log_dir = "/home/mohammad/cmg-x/experiments/logs/gnn"
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/{args.dataset}_cmg_{args.seed}.json"
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)

    fig_base = os.path.join(log_dir, f"{args.dataset}_cmg_{args.seed}")
    plt.figure()
    plt.plot(logs['train_acc'], label='Train Acc')
    plt.plot(logs['val_acc'], label='Val Acc')
    plt.plot(logs['test_acc'], label='Test Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(fig_base + '_accuracy.png')
