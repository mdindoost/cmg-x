import os
import networkx as nx
from networkx.generators.community import stochastic_block_model

output_dir = "experiments/data/synthetic"
os.makedirs(output_dir, exist_ok=True)

def save_graph(g, name):
    path = os.path.join(output_dir, f"{name}.edgelist")
    nx.write_edgelist(g, path, data=False)
    print(f"✅ Saved: {path}")

# GRID: 5x5, 10x10, 20x20
save_graph(nx.convert_node_labels_to_integers(nx.grid_2d_graph(5, 5)), "grid_small")
save_graph(nx.convert_node_labels_to_integers(nx.grid_2d_graph(10, 10)), "grid_medium")
save_graph(nx.convert_node_labels_to_integers(nx.grid_2d_graph(20, 20)), "grid_large")

# PATH: 20, 100, 500
save_graph(nx.path_graph(20), "path_small")
save_graph(nx.path_graph(100), "path_medium")
save_graph(nx.path_graph(500), "path_large")

# PATH + STAR: hybrid graphs
def path_star(n_path, n_arms):
    g = nx.path_graph(n_path)
    star = nx.star_graph(n_arms)
    offset = max(g.nodes) + 1
    star = nx.relabel_nodes(star, lambda x: x + offset)
    g.add_nodes_from(star.nodes)
    g.add_edges_from(star.edges)
    g.add_edge(n_path // 2, offset)  # Connect center of star to mid path
    return g

save_graph(path_star(20, 3), "pathstar_small")
save_graph(path_star(50, 5), "pathstar_medium")
save_graph(path_star(100, 10), "pathstar_large")

# SBM: 4 blocks with intra/inter probabilities
def sbm_graph(size_per_block):
    sizes = [size_per_block] * 4
    p_intra = 0.6
    p_inter = 0.02
    probs = [[p_intra if i == j else p_inter for j in range(4)] for i in range(4)]
    return stochastic_block_model(sizes, probs, seed=42)

save_graph(sbm_graph(10), "sbm_small")
save_graph(sbm_graph(25), "sbm_medium")
save_graph(sbm_graph(50), "sbm_large")

# ER graphs: G(n, p)
save_graph(nx.erdos_renyi_graph(30, 0.1, seed=42), "er_small")
save_graph(nx.erdos_renyi_graph(100, 0.05, seed=42), "er_medium")
save_graph(nx.erdos_renyi_graph(300, 0.02, seed=42), "er_large")
