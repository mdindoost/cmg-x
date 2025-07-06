
import os
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from cmgx.pyg_pool import CMGPooling
from cmgx.torch_interface import cmg_unpool_features
import matplotlib.pyplot as plt


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


class GNNWithCMG(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool = CMGPooling()
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, batch, P, L = self.pool(x, edge_index, batch=batch, return_all=True)
        x_coarse = x.clone()
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x), P, x_coarse


def compute_metrics(data, P_list, X_coarse_all):
    x_fine_parts = []
    cursor = 0
    cluster_sizes = []

    for i, P in enumerate(P_list):
        C = P.shape[1]
        cluster_sizes.append(C)
        X_c = X_coarse_all[cursor:cursor + C]

        if X_c.shape[1] != P.shape[1]:
            X_c = X_c[:, :P.shape[1]] if X_c.shape[1] > P.shape[1] else F.pad(X_c, (0, P.shape[1] - X_c.shape[1]))

        x_fine = P @ X_c
        if i == 0:
            F_shape = x_fine.shape[1]
        else:
            x_fine = x_fine[:, :F_shape] if x_fine.shape[1] > F_shape else F.pad(x_fine, (0, F_shape - x_fine.shape[1]))

        x_fine_parts.append(x_fine)
        cursor += C

    x_fine = torch.cat(x_fine_parts, dim=0)
    data_x = data.x[:, :x_fine.shape[1]] if data.x.shape[1] > x_fine.shape[1] else F.pad(data.x, (0, x_fine.shape[1] - data.x.shape[1]))

    cos_sim = F.cosine_similarity(data_x, x_fine, dim=-1).mean().item()
    mse = F.mse_loss(data_x, x_fine).item()
    compression = (X_coarse_all.size(0) / data.x.size(0))
    return cos_sim, mse, compression, cluster_sizes


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

    # if dataset.num_node_features == 0:
    #     for data in dataset:
    #         num_nodes = data.num_nodes
    #         data.x = torch.ones((num_nodes, 1))


    num_classes = dataset.num_classes
    num_features = dataset.num_features

    split = int(0.8 * len(dataset))
    train_dataset = dataset[:split]
    test_dataset = dataset[split:]
    val_split = int(0.1 * split)
    val_dataset = train_dataset[:val_split]
    train_dataset = train_dataset[val_split:]

    # ✅ Patch ALL graphs used
    all_graphs = train_dataset + val_dataset + test_dataset
    for data in all_graphs:
        if data.x is None:
            data.x = torch.ones((data.num_nodes, 1))
        
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = GNNWithCMG(num_features, 64, num_classes).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logs = {'train_acc': [], 'val_acc': [], 'test_acc': [], 'compression': [], 'cos_sim': [], 'mse': [], 'cluster_stats': []}
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

    os.makedirs('/home/mohammad/cmg-x/experiments/logs/gnn', exist_ok=True)
    log_path = f"/home/mohammad/cmg-x/experiments/logs/gnn/{args.dataset}_cmg_{args.seed}.json"
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)

    fig_base = os.path.join('/home/mohammad/cmg-x/experiments/logs/gnn', f"{args.dataset}_cmg_{args.seed}")
    plt.figure()
    plt.plot(logs['train_acc'], label='Train Acc')
    plt.plot(logs['val_acc'], label='Val Acc')
    plt.plot(logs['test_acc'], label='Test Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(fig_base + '_accuracy.png')

    plt.figure()
    plt.plot(logs['compression'], label='Compression')
    plt.title('Compression Ratio over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Compression')
    plt.legend()
    plt.savefig(fig_base + '_compression.png')

    plt.figure()
    plt.plot(logs['cos_sim'], label='Cosine Similarity')
    plt.plot(logs['mse'], label='MSE')
    plt.title('Feature Reconstruction Quality')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(fig_base + '_reconstruction.png')


if __name__ == '__main__':
    main()
