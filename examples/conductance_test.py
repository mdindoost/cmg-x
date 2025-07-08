#!/usr/bin/env python3
"""
Test the fixed conductance calculations and CMG integration.
"""

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import from_networkx, to_scipy_sparse_matrix
import networkx as nx
from collections import Counter

# Import your modules (adjust paths as needed)
try:
    from cmgx.core import cmgCluster
    from cmgx.filtered import (
        evaluate_phi_conductance, 
        cmg_filtered_clustering,
        compute_lambda_critical
    )
    CMGX_AVAILABLE = True
except ImportError as e:
    print(f"❌ CMG-X import failed: {e}")
    CMGX_AVAILABLE = False

def create_test_graph_simple():
    """Create simple test graph: two triangles + bridge."""
    print("🔧 Creating simple test graph (two triangles + bridge)")
    
    # Two triangles connected by weak bridge
    G = nx.Graph()
    
    # Triangle 1: nodes 0, 1, 2
    G.add_edges_from([(0, 1, {'weight': 2.0}), 
                      (1, 2, {'weight': 2.0}), 
                      (2, 0, {'weight': 2.0})])
    
    # Triangle 2: nodes 3, 4, 5
    G.add_edges_from([(3, 4, {'weight': 2.0}), 
                      (4, 5, {'weight': 2.0}), 
                      (5, 3, {'weight': 2.0})])
    
    # Weak bridge
    G.add_edge(1, 4, weight=0.1)
    
    return G

def test_adjacency_matrix_access():
    """Test that we can properly access adjacency matrix neighbors."""
    print("\n🔍 Testing adjacency matrix neighbor access")
    
    G = create_test_graph_simple()
    data = from_networkx(G)
    
    # Build adjacency matrix
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    
    # Handle edge weights if present
    if hasattr(data, 'edge_weight') and data.edge_weight is not None:
        # Rebuild with weights
        rows, cols = data.edge_index[0].numpy(), data.edge_index[1].numpy()
        weights = data.edge_weight.numpy()
        A = sp.coo_matrix((weights, (rows, cols)), shape=(data.num_nodes, data.num_nodes))
    
    A = A.tocsr()  # Convert to CSR for row access
    
    print(f"Adjacency matrix: {A.shape}, format: {A.format}, nnz: {A.nnz}")
    
    # Test neighbor access for each node
    for node in range(A.shape[0]):
        row_start = A.indptr[node]
        row_end = A.indptr[node + 1]
        
        neighbors = A.indices[row_start:row_end]
        weights = A.data[row_start:row_end]
        
        print(f"Node {node}: neighbors {neighbors.tolist()}, weights {weights.tolist()}")
    
    return A, data

def test_conductance_calculation_methods():
    """Test both conductance calculation methods."""
    print("\n📊 Testing conductance calculation methods")
    
    A, data = test_adjacency_matrix_access()
    
    # Test with perfect clustering (should have low conductance)
    perfect_labels = np.array([0, 0, 0, 1, 1, 1])  # Two triangles
    print(f"\nTesting perfect clustering: {perfect_labels}")
    
    # Test both methods
    phi_both = evaluate_phi_conductance(A, perfect_labels, method='both')
    
    print("Results:")
    print(f"  Standard conductance (avg): {phi_both['avg_phi']:.4f}")
    print(f"  Normalized conductance (avg): {phi_both['normalized_avg_phi']:.4f}")
    print(f"  Ratio (norm/std): {phi_both['comparison']['ratio']:.4f}")
    
    print("\nPer-cluster details:")
    for cid in sorted(phi_both['phi'].keys()):
        std_phi = phi_both['phi'][cid]
        norm_phi = phi_both['normalized_phi'][cid]
        print(f"  Cluster {cid}: φ_std={std_phi:.4f}, φ_norm={norm_phi:.4f}")
    
    # Test with bad clustering (should have high conductance)
    bad_labels = np.array([0, 1, 0, 1, 0, 1])  # Alternating
    print(f"\nTesting bad clustering: {bad_labels}")
    
    phi_bad = evaluate_phi_conductance(A, bad_labels, method='both')
    
    print("Bad clustering results:")
    print(f"  Standard conductance (avg): {phi_bad['avg_phi']:.4f}")
    print(f"  Normalized conductance (avg): {phi_bad['normalized_avg_phi']:.4f}")
    
    return A, perfect_labels, bad_labels

def test_cmg_integration():
    """Test CMG integration with conductance calculation."""
    print("\n⚙️  Testing CMG integration")
    
    if not CMGX_AVAILABLE:
        print("❌ CMG-X not available, skipping CMG test")
        return
    
    A, data = test_adjacency_matrix_access()
    
    # Test baseline CMG
    print("\n--- Baseline CMG ---")
    degrees = np.array(A.sum(axis=1)).flatten()
    L = sp.diags(degrees) - A
    
    try:
        cI_raw, nc = cmgCluster(L.tocsc())
        cI = cI_raw - 1  # Convert to 0-indexed
        
        print(f"CMG result: {nc} clusters")
        print(f"Raw output (1-indexed): {cI_raw}")
        print(f"Converted (0-indexed): {cI}")
        print(f"Cluster sizes: {dict(Counter(cI))}")
        
        # Test conductance on CMG result
        phi_cmg = evaluate_phi_conductance(A, cI, method='both')
        print(f"CMG conductance - Standard: {phi_cmg['avg_phi']:.4f}, "
              f"Normalized: {phi_cmg['normalized_avg_phi']:.4f}")
        
    except Exception as e:
        print(f"❌ Baseline CMG failed: {e}")
        return
    
    # Test filtered CMG
    print("\n--- Filtered CMG ---")
    try:
        cI_filt, nc_filt, phi_filt, lambda_crit = cmg_filtered_clustering(
            data, k=10, d=20, threshold=0.1, conductance_method='both'
        )
        
        print(f"Filtered CMG result: {nc_filt} clusters")
        print(f"λ_critical: {lambda_crit:.4f}")
        print(f"Cluster assignment: {cI_filt}")
        print(f"Cluster sizes: {dict(Counter(cI_filt))}")
        
        if 'comparison' in phi_filt:
            print(f"Filtered conductance - Standard: {phi_filt['comparison']['standard_avg']:.4f}, "
                  f"Normalized: {phi_filt['comparison']['normalized_avg']:.4f}")
        else:
            print(f"Filtered conductance: {phi_filt['avg_phi']:.4f}")
        
    except Exception as e:
        print(f"❌ Filtered CMG failed: {e}")

def test_filter_parameters():
    """Test different filter parameters."""
    print("\n🎛️  Testing filter parameters")
    
    if not CMGX_AVAILABLE:
        print("❌ CMG-X not available, skipping filter test")
        return
    
    A, data = test_adjacency_matrix_access()
    
    k_values = [2, 5, 10, 15]
    
    print("Filter parameter analysis:")
    print("k\tλ_crit\tClusters\tStd_φ\t\tNorm_φ")
    print("-" * 55)
    
    for k in k_values:
        try:
            lambda_crit = compute_lambda_critical(k)
            cI, nc, phi_stats, _ = cmg_filtered_clustering(
                data, k=k, d=20, threshold=0.1, conductance_method='both'
            )
            
            if 'comparison' in phi_stats:
                std_phi = phi_stats['comparison']['standard_avg']
                norm_phi = phi_stats['comparison']['normalized_avg']
            else:
                std_phi = phi_stats['avg_phi']
                norm_phi = std_phi  # Fallback
            
            std_str = f"{std_phi:.4f}" if std_phi != float('inf') else "inf"
            norm_str = f"{norm_phi:.4f}" if norm_phi != float('inf') else "inf"
            
            print(f"{k}\t{lambda_crit:.4f}\t{nc}\t\t{std_str}\t\t{norm_str}")
            
        except Exception as e:
            print(f"{k}\tFAILED: {str(e)[:30]}...")

def test_large_example():
    """Test on a larger, more complex graph."""
    print("\n🏗️  Testing on larger graph")
    
    if not CMGX_AVAILABLE:
        print("❌ CMG-X not available, skipping large test")
        return
    
    # Create graph with 3 cliques + bridges
    G = nx.Graph()
    
    # Clique 1: nodes 0-3
    clique1 = list(range(4))
    for i in clique1:
        for j in clique1:
            if i < j:
                G.add_edge(i, j, weight=1.0)
    
    # Clique 2: nodes 4-7
    clique2 = list(range(4, 8))
    for i in clique2:
        for j in clique2:
            if i < j:
                G.add_edge(i, j, weight=1.0)
    
    # Clique 3: nodes 8-11
    clique3 = list(range(8, 12))
    for i in clique3:
        for j in clique3:
            if i < j:
                G.add_edge(i, j, weight=1.0)
    
    # Weak bridges
    G.add_edge(1, 5, weight=0.05)  # Bridge 1-2
    G.add_edge(6, 9, weight=0.05)  # Bridge 2-3
    
    data = from_networkx(G)
    
    # Convert edge weights
    edge_weights = [G[u.item()][v.item()]['weight'] for u, v in data.edge_index.t()]
    data.edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    
    print(f"Large graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
    print("Expected: 3 clusters corresponding to cliques")
    
    try:
        cI, nc, phi_stats, lambda_crit = cmg_filtered_clustering(
            data, k=10, d=30, threshold=0.1, conductance_method='both'
        )
        
        print(f"Result: {nc} clusters found")
        print(f"Cluster assignment: {cI}")
        print(f"Cluster sizes: {dict(Counter(cI))}")
        
        if 'comparison' in phi_stats:
            print(f"Conductance - Standard: {phi_stats['comparison']['standard_avg']:.4f}, "
                  f"Normalized: {phi_stats['comparison']['normalized_avg']:.4f}")
        
        # Check if clustering is reasonable
        if nc == 3:
            print("✅ Found expected number of clusters!")
        else:
            print(f"⚠️  Expected 3 clusters, got {nc}")
        
    except Exception as e:
        print(f"❌ Large graph test failed: {e}")

def run_all_tests():
    """Run all conductance and CMG tests."""
    print("🧪 Comprehensive CMG-X Conductance Tests")
    print("=" * 60)
    
    try:
        # Basic tests
        test_adjacency_matrix_access()
        test_conductance_calculation_methods()
        
        if CMGX_AVAILABLE:
            test_cmg_integration()
            test_filter_parameters()
            test_large_example()
        else:
            print("\n❌ CMG-X not available - skipping CMG-specific tests")
            print("Install CMG-X to run complete tests")
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
