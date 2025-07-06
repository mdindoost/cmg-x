import os
import glob
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import json
from scipy.sparse import csc_matrix
from utils_structure_eval import compute_path_length_distortion
import time

GRAPH_DIR = "experiments/data/synthetic"
LOG_DIR = "experiments/logs/structure_eval_graphzoom"
os.makedirs(LOG_DIR, exist_ok=True)

def run_graphzoom_clustering(graph: nx.Graph, max_k: int = 50):
    A = csc_matrix(nx.to_scipy_sparse_array(graph, dtype=np.float64))
    L = csgraph.laplacian(A, normed=True)

    n_nodes = A.shape[0]
    k = min(max_k, n_nodes - 1)

    eigval, eigvec = eigsh(L, k=k, which='SM')
    embeddings = normalize(eigvec)

    k_cluster = max(2, n_nodes // 10)  # Estimate clusters
    kmeans = KMeans(n_clusters=k_cluster, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels, k_cluster

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
    plt.title("GraphZoom Cluster Size Histogram")
    plt.xlabel("Cluster Size")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def compute_conductance(graph, labels):
    cuts = sum(1 for u, v in graph.edges if labels[u] != labels[v])
    return cuts / graph.number_of_edges()

def print_and_save_metrics(graph, labels, n_clusters, log_dir, distortion, runtime):
    _, sizes = np.unique(labels, return_counts=True)
    conductance = compute_conductance(graph, labels)
    metrics = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_clusters": int(n_clusters),
        "cluster_sizes": sizes.tolist(),
        "conductance": conductance,
        "path_length_distortion": round(distortion, 4),
        "runtime_seconds": round(runtime, 4)
    }
    print(json.dumps(metrics, indent=2))
    with open(os.path.join(log_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


# Run on all synthetic graphs
graph_files = sorted(glob.glob(os.path.join(GRAPH_DIR, "*.edgelist")))

for path in graph_files:
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"\n📊 Running GraphZoom on: {name}")

    G = nx.read_edgelist(path, nodetype=int)
    G = nx.convert_node_labels_to_integers(G)

    start = time.time()
    labels, k = run_graphzoom_clustering(G)
    runtime = time.time() - start
    distortion = compute_path_length_distortion(G, labels)

    out_dir = os.path.join(LOG_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    plot_colored_clusters(G, labels, f"GraphZoom Clusters: {name}", os.path.join(out_dir, "clusters.png"))
    cluster_histogram(labels, os.path.join(out_dir, "cluster_histogram.png"))
    print_and_save_metrics(G, labels, k, out_dir, distortion, runtime)
