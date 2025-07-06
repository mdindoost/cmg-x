import os
import json
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import csc_matrix, identity
from cmgx import cmgCluster

LOG_DIR = "experiments/logs/structure_eval"
DATA_DIR = "experiments/data/synthetic"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def run_cmg_clustering(graph: nx.Graph):
    A = csc_matrix(nx.to_scipy_sparse_array(graph, dtype=np.float64))
    L = sp.csgraph.laplacian(A, normed=False).astype(np.float64)
    L += 1e-8 * identity(L.shape[0], format='csc')

    start = time.time()
    labels, n_clusters = cmgCluster(L)
    runtime = time.time() - start

    labels = labels - 1  # shift to 0-based
    return labels, n_clusters, runtime

def plot_colored_clusters(graph, labels, title, out_path):
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(5, 5))
    nx.draw_networkx_nodes(graph, pos, node_color=labels, cmap='tab20', node_size=100)
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def cluster_histogram(labels, out_path):
    _, counts = np.unique(labels, return_counts=True)
    plt.figure()
    plt.hist(counts, bins=range(1, max(counts)+2), edgecolor='black')
    plt.title("Cluster Size Histogram")
    plt.xlabel("Cluster Size")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def compute_conductance(graph, labels):
    return sum(1 for u, v in graph.edges if labels[u] != labels[v]) / graph.number_of_edges()

def compute_path_length_distortion(graph, labels):
    try:
        orig_avg = nx.average_shortest_path_length(graph)
    except:
        return None
    cluster_graph = nx.Graph()
    cluster_ids = np.unique(labels)
    cluster_graph.add_nodes_from(cluster_ids)
    for u, v in graph.edges():
        cu, cv = labels[u], labels[v]
        if cu != cv:
            cluster_graph.add_edge(cu, cv)
    try:
        cluster_avg = nx.average_shortest_path_length(cluster_graph)
    except:
        return None
    return round(cluster_avg / orig_avg, 4)

def print_and_save_metrics(graph, labels, n_clusters, log_dir, runtime):
    _, sizes = np.unique(labels, return_counts=True)
    conductance = compute_conductance(graph, labels)
    distortion = compute_path_length_distortion(graph, labels)
    metrics = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_clusters": int(n_clusters),
        "cluster_sizes": sizes.tolist(),
        "conductance": conductance,
        "path_length_distortion": distortion,
        "runtime_seconds": round(runtime, 4)
    }
    print(json.dumps(metrics, indent=2))
    with open(os.path.join(log_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
