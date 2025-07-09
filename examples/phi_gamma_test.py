import numpy as np
import torch
from torch_geometric.data import Data
from cmgx.filtered import evaluate_phi_conductance  # Assuming you updated this

# Define edges
edges = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 9]
    # [7, 8]
]).T  # Shape: [2, E]

# Create edge_index tensor
edge_index = torch.tensor(edges, dtype=torch.long)

# Define number of nodes
num_nodes = 10

# Create PyG data object
data = Data(edge_index=edge_index, num_nodes=num_nodes)

# Cluster labels
# Cluster 0 = {0, 1, 3, 4}, rest = Cluster 1
labels = np.ones(num_nodes, dtype=int)
labels[[0, 1, 2, 3]] = 0

# Evaluate φ, γ, φ·γ
results = evaluate_phi_conductance(data, labels)

# Print result
from pprint import pprint
pprint(results)
