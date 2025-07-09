import os
import json
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

def load_latest_log(dataset, log_dir="logs/structure_eval/"):
    logs = sorted([
        f for f in os.listdir(log_dir)
        if f.endswith(".json") and dataset in f and "config" in open(os.path.join(log_dir, f)).read()
    ], key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)

    if not logs:
        raise FileNotFoundError(f"No logs found for dataset {dataset}")

    with open(os.path.join(log_dir, logs[0]), "r") as f:
        data = json.load(f)
    print(f"[LOADED] {logs[0]}")
    return data

def plot_clusters(G, cluster_labels, title, save_path):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    cmap = plt.get_cmap("tab20")

    unique_clusters = sorted(set(cluster_labels))
    color_map = {cid: cmap(i % 20) for i, cid in enumerate(unique_clusters)}

    node_colors = [color_map[cid] for cid in cluster_labels]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=30, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[SAVED] {save_path}")

def main():
    dataset = "Cora"  # change to Citeseer or Pubmed as needed
    data = Planetoid(root=f"data/{dataset}", name=dataset)[0]
    G = to_networkx(data, to_undirected=True)

    log = load_latest_log(dataset)
    labels = log.get("cluster_labels")
    if labels is None:
        raise ValueError("No cluster_labels found in log")

    if isinstance(labels, list):
        labels = [int(x) for x in labels]
    elif isinstance(labels, torch.Tensor):
        labels = labels.tolist()

    config = log["config"]
    save_dir = "logs/structure_eval/plots"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{dataset}_k{config['k']}_cluster_coloring.png"
    title = f"CMG Clustering ({dataset}) | k={config['k']}, d={config['d']}, t={config['threshold']}"
    save_path = os.path.join(save_dir, fname)

    plot_clusters(G, labels, title, save_path)

if __name__ == "__main__":
    main()
