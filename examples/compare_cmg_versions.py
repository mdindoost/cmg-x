import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from cmgx.core import cmgCluster
from torch_geometric.utils import to_scipy_sparse_matrix, from_networkx
from scipy.linalg import qr, eigh
from collections import defaultdict, Counter
import networkx as nx
import pandas as pd
import os
from cmgx.filtered import cmg_filtered_clustering, evaluate_phi_conductance




def run_baseline_cmg(data):
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()  # Use CSR for correct row access
    A_csr = A.tocsr()  # Fix: ensure correct format for row-wise access
    L = sp.diags(A.sum(axis=1).A.flatten()) - A  # L is OK in CSR here
    cI, nc = cmgCluster(L.tocsc())  # Convert to CSC only for CMG
    # print("cI= ", cI)
    phi = evaluate_phi_conductance(A_csr, cI)
    return cI, nc, phi, A

def print_cluster_sizes(cI):
    counts = Counter(cI)
    return ", ".join([f"cluster {k}: {v} nodes" for k, v in sorted(counts.items())])

def save_edge_list(matrix, filepath):
    coo = matrix.tocoo()
    with open(filepath, 'w') as f:
        for u, v, w in zip(coo.row, coo.col, coo.data):
            f.write(f"{u}\t{v}\t{w:.4f}\n")

def save_edge_matrix(matrix, filepath):
    coo = matrix.tocoo()
    np.savetxt(filepath, np.vstack((coo.row, coo.col)), fmt='%d')

def save_cluster_edges(matrix, labels, filepath_prefix):
    coo = matrix.tocoo()
    edges_by_cluster = defaultdict(list)
    for u, v, w in zip(coo.row, coo.col, coo.data):
        if labels[u] == labels[v]:
            edges_by_cluster[labels[u]].append((u, v, w))

    for cluster_id, edges in edges_by_cluster.items():
        with open(f"{filepath_prefix}_cluster_{cluster_id}_edges.tsv", 'w') as f:
            for u, v, w in edges:
                f.write(f"{u}\t{v}\t{w:.4f}\n")
        with open(f"{filepath_prefix}_cluster_{cluster_id}_adj_matrix.txt", 'w') as f:
            rows = [u for u, v, w in edges]
            cols = [v for u, v, w in edges]
            np.savetxt(f, np.vstack((rows, cols)), fmt='%d')

def compare_on_graph(name, data):
    print(f"\n===== Testing on: {name} =====")
    os.makedirs("logs", exist_ok=True)
    csv_path = f"logs/{name.replace(' ', '_')}_summary.csv"
    edge_path_base = f"logs/{name.replace(' ', '_')}"

    # Baseline CMG
    print("[BASELINE CMGX]")
    cI_base, nc_base, phi_base, A_base = run_baseline_cmg(data)
    print("CMGX Clusters = ", cI_base)
    print("CMGX #Clusters = ", nc_base)
    print("[DEBUG] CMGX φ per cluster:")
    for cid, phi in sorted(phi_base['phi'].items()):
        print(f"  Cluster {cid}: φ = {phi:.4f}")

    save_edge_list(A_base, f"{edge_path_base}_original_edges.tsv")
    save_edge_matrix(A_base, f"{edge_path_base}_original_adj_matrix.txt")
    save_cluster_edges(A_base, cI_base, f"{edge_path_base}_CMGX")

    # Filtered CMG
    print("[FILTERED PIPELINE]")
    cI_filt, nc_filt, phi_filt = cmg_filtered_clustering(data)
    print("FILTERED Clusters = ", cI_filt)
    print("FILTERED #Clusters = ", nc_filt)
    print("[DEBUG] Filtered φ per cluster:")
    for cid, phi in sorted(phi_filt['phi'].items()):
        print(f"  Cluster {cid}: φ = {phi:.4f}")

    A_filt = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsc()
    save_cluster_edges(A_filt, cI_filt, f"{edge_path_base}_Filtered")

    # Results Table
    df = pd.DataFrame({
        "Method": ["CMGX", "Filtered"],
        "#Clusters": [nc_base, nc_filt],
        "AvgPhi": [phi_base['avg_phi'], phi_filt['avg_phi']]
    })
    print(df.to_markdown(index=False))
    df.to_csv(csv_path, index=False)

def create_path_graph(n):
    G = nx.path_graph(n)
    return from_networkx(G)

def create_grid_graph(n):
    G = nx.grid_2d_graph(n, n)
    mapping = {(i, j): i * n + j for i in range(n) for j in range(n)}
    G = nx.relabel_nodes(G, mapping)
    return from_networkx(G)

def create_er_graph(n, p, seed=42):
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() == 0:
        raise RuntimeError("Empty ER graph generated")
    return from_networkx(G)

def main():
    path_data = create_path_graph(10)
    compare_on_graph("Path Graph (10)", path_data)
    
    # Grid 10x10
    grid_data = create_grid_graph(10)
    compare_on_graph("2D Grid (10x10)", grid_data)

    # ER graphs (10, 0.2), (50, 0.1), (100, 0.05)
    for n, p in [(10, 0.2), (50, 0.1), (100, 0.05)]:
        try:
            er_data = create_er_graph(n, p)
            compare_on_graph(f"ER Graph (n={n}, p={p})", er_data)
        except RuntimeError as e:
            print(f"[SKIPPED] ER Graph (n={n}, p={p}): {e}")
if __name__ == '__main__':
    main()
