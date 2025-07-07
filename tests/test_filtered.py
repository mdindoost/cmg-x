import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from cmgx.filtered import cmg_filtered_clustering

def test_two_clusters():
    # Create a graph with 2 strong clusters joined by weak edge
    G = nx.Graph()

    # Cluster 1: nodes 0-4, fully connected
    G.add_edges_from([(i, j) for i in range(5) for j in range(i+1, 5)])
    
    # Cluster 2: nodes 5-9, fully connected
    G.add_edges_from([(i, j) for i in range(5, 10) for j in range(i+1, 10)])

    # Weak connection between clusters
    G.add_edge(2, 7, weight=0.05)

    # Add default weights if missing
    for u, v in G.edges():
        if 'weight' not in G[u][v]:
            G[u][v]['weight'] = 1.0

    # Convert to PyG format
    data = from_networkx(G)

    # Run CMG + filtering
    labels, n_clusters, phi = cmg_filtered_clustering(data, k=10, d=10, threshold=0.1)

    print(f"[TEST] Cluster labels: {labels}")
    print(f"[TEST] Number of clusters: {n_clusters}")
    print(f"[TEST] φγ Conductance: {phi['avg_conductance']:.4f}")
    assert n_clusters <= 3, "Expected CMG to detect ~2-3 clusters"
    assert phi['avg_conductance'] < 0.5, "Expected good φγ conductance"

if __name__ == '__main__':
    test_two_clusters()
