import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

def generate_synthetic_graphs(save_dir="experiments/data/synthetic"):
    os.makedirs(save_dir, exist_ok=True)

    graphs = {
        "chain_20": nx.path_graph(20),
        "grid_5x5": nx.grid_2d_graph(5, 5),
        "tree_3_3": nx.balanced_tree(r=3, h=3)
    }

    pyg_graphs = {}

    for name, G in graphs.items():
        # Relabel grid nodes to integer
        if name == "grid_5x5":
            G = nx.convert_node_labels_to_integers(G)

        # Convert to PyG Data
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # undirected
        x = torch.eye(G.number_of_nodes(), dtype=torch.float)  # identity features
        data = Data(x=x, edge_index=edge_index)
        pyg_graphs[name] = data

        # Plot and save PNG
        plt.figure(figsize=(4, 4))
        nx.draw(G, with_labels=True, node_size=300, node_color='skyblue')
        plt.title(name)
        plt.savefig(os.path.join(save_dir, f"{name}.png"))
        plt.close()

    # Save all graphs as .pt
    torch.save(pyg_graphs, os.path.join(save_dir, "synthetic_graphs.pt"))
    print(f"✅ Saved synthetic graphs and images to: {save_dir}")

if __name__ == "__main__":
    generate_synthetic_graphs()
