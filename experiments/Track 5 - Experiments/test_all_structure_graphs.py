import os
import glob
import networkx as nx
from utils_structure_eval import *

GRAPH_DIR = "experiments/data/synthetic"
LOG_DIR = "experiments/logs/structure_eval"
os.makedirs(LOG_DIR, exist_ok=True)

graph_files = sorted(glob.glob(os.path.join(GRAPH_DIR, "*.edgelist")))

for path in graph_files:
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"\n📊 Running CMG on: {name}")

    G = nx.read_edgelist(path, nodetype=int)
    G = nx.convert_node_labels_to_integers(G)

    labels, n_clusters, runtime = run_cmg_clustering(G)

    out_dir = os.path.join(LOG_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    plot_colored_clusters(G, labels, f"CMG Clusters: {name}", os.path.join(out_dir, "clusters.png"))
    cluster_histogram(labels, os.path.join(out_dir, "cluster_histogram.png"))
    print_and_save_metrics(G, labels, n_clusters, out_dir, runtime)
