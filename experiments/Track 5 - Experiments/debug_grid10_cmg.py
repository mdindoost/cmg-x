import networkx as nx
import numpy as np
from scipy.sparse import csgraph, csc_matrix, identity
from cmgx import cmgCluster

# Build 10x10 grid
G = nx.grid_2d_graph(10, 10)
G = nx.convert_node_labels_to_integers(G)

# Adjacency matrix (CSC, float64)
A = csc_matrix(nx.to_scipy_sparse_array(G, dtype=np.float64))
print(f"[DEBUG] A shape={A.shape}, nnz={A.nnz}, dtype={A.dtype}")

# Laplacian (float64) + tiny diagonal to avoid divide-by-zero
L = csgraph.laplacian(A, normed=False).astype(np.float64)
L += 1e-8 * identity(L.shape[0], format='csc')

print(f"[DEBUG] L shape={L.shape}, nnz={L.nnz}, dtype={L.dtype}")
print(f"[DEBUG] L sample data: {L.data[:10]}")

# Run CMG
labels, n_clusters = cmgCluster(L)
labels = labels -1 
unique_labels = np.unique(labels)

# Report
print(f"\n[RESULT] CMG returned {n_clusters} clusters")
print(f"[RESULT] Unique cluster IDs: {unique_labels}")
print(f"[RESULT] Cluster counts: {np.bincount(labels)}")
