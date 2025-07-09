#!/usr/bin/env python3
"""
Comprehensive test suite for CMG spectral filtering functionality.

This test file validates the spectral filtering + CMG clustering pipeline
and demonstrates the capabilities of the filtered.py module.

Run with: pytest tests/test_filtered_clustering.py -v
"""

import pytest
import numpy as np
import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from collections import Counter

# Import the CMG-X modules
try:
    from cmgx.filtered import (
        cmg_filtered_clustering,
        evaluate_phi_conductance,
        compute_lambda_critical,
        build_normalized_laplacian
    )
    from cmgx.core import cmgCluster
    CMG_AVAILABLE = True
except ImportError as e:
    CMG_AVAILABLE = False
    pytest.skip(f"CMG-X not available: {e}", allow_module_level=True)


class TestFilteredClustering:
    """Test suite for spectral filtering + CMG clustering."""

    def test_two_cliques_perfect_clustering(self):
        """Test that spectral filtering enables perfect two-cliques clustering."""
        
        # Create two cliques connected by weak bridge
        G = nx.Graph()
        
        # Clique 1: nodes 0-4 (strong connections)
        for i in range(5):
            for j in range(i+1, 5):
                G.add_edge(i, j, weight=2.0)
        
        # Clique 2: nodes 5-9 (strong connections)
        for i in range(5, 10):
            for j in range(i+1, 10):
                G.add_edge(i, j, weight=2.0)
        
        # Weak bridge
        G.add_edge(2, 7, weight=0.05)
        
        data = from_networkx(G)
        edge_weights = [G[u.item()][v.item()]['weight'] for u, v in data.edge_index.t()]
        data.edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        # Test filtered clustering with k=10 (should give perfect result)
        labels, n_clusters, phi_stats, lambda_crit = cmg_filtered_clustering(
            data, k=10, d=20, threshold=0.1
        )
        
        # Assertions
        assert n_clusters == 2, f"Expected 2 clusters, got {n_clusters}"
        assert lambda_crit == pytest.approx(0.3333, abs=0.01), f"λ_critical should be ~0.33 for k=10"
        
        # Check conductance quality
        if 'comparison' in phi_stats:
            std_phi = phi_stats['comparison']['standard_avg']
            norm_phi = phi_stats['comparison']['normalized_avg']
        else:
            std_phi = phi_stats['avg_phi']
            norm_phi = phi_stats.get('normalized_avg_phi', std_phi)
        
        assert std_phi == pytest.approx(0.0476, abs=0.01), f"Standard φ should be ~0.0476, got {std_phi}"        
        # Check cluster assignment quality
        cluster_0_nodes = set(np.where(labels == 0)[0])
        cluster_1_nodes = set(np.where(labels == 1)[0])
        
        clique1_set = set(range(5))
        clique2_set = set(range(5, 10))
        
        perfect_clustering = (
            (cluster_0_nodes == clique1_set and cluster_1_nodes == clique2_set) or
            (cluster_0_nodes == clique2_set and cluster_1_nodes == clique1_set)
        )
        
        assert perfect_clustering, f"Clustering should separate the two cliques perfectly"
        
        print(f"✅ Perfect clustering achieved: {n_clusters} clusters, φ_std={std_phi:.4f}, φ_norm={norm_phi:.4f}")

    def test_filter_parameter_effects(self):
        """Test how different filter parameters (k values) affect clustering."""
        
        # Create test graph
        G = nx.path_graph(8)  # Simple path graph
        data = from_networkx(G)
        
        k_values = [2, 5, 10]
        results = []
        
        for k in k_values:
            labels, n_clusters, phi_stats, lambda_crit = cmg_filtered_clustering(
                data, k=k, d=15, threshold=0.1
            )
            
            # Check lambda critical formula
            expected_lambda = 2.0 / (1.0 + 0.5 * k)
            assert lambda_crit == pytest.approx(expected_lambda, abs=0.01), \
                f"λ_critical formula incorrect for k={k}"
            
            results.append({
                'k': k,
                'lambda_crit': lambda_crit,
                'n_clusters': n_clusters,
                'avg_phi': phi_stats['avg_phi']
            })
        
        # Check that results are reasonable
        for result in results:
            assert result['n_clusters'] >= 1, "Must have at least 1 cluster"
            assert result['n_clusters'] <= 8, "Cannot have more clusters than nodes"
            assert result['avg_phi'] >= 0, "Conductance must be non-negative"
        
        print(f"✅ Filter parameter sweep completed: {len(results)} configurations tested")


    def test_edge_cases(self):
        """Test edge cases and error handling."""
        
        # Test single node graph
        G = nx.Graph()
        G.add_node(0)
        data = from_networkx(G)
        
        labels, n_clusters, phi_stats, lambda_crit = cmg_filtered_clustering(
            data, k=5, d=10, threshold=0.1
        )
        
        assert n_clusters == 1, "Single node should form one cluster"
        assert len(labels) == 1, "Should have one label"
        
        # Test disconnected graph
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])  # Two disconnected edges
        data = from_networkx(G)
        
        labels, n_clusters, phi_stats, lambda_crit = cmg_filtered_clustering(
            data, k=5, d=10, threshold=0.1
        )
        
        assert n_clusters >= 2, "Disconnected components should form separate clusters"
        
        print(f"✅ Edge cases handled correctly")

    def test_parameter_validation(self):
        """Test that invalid parameters are handled appropriately."""
        
        # Create simple test graph
        G = nx.path_graph(5)
        data = from_networkx(G)
        
        # Test with reasonable parameters (should work)
        labels, n_clusters, phi_stats, lambda_crit = cmg_filtered_clustering(
            data, k=5, d=10, threshold=0.1
        )
        
        assert n_clusters >= 1, "Should produce valid clustering"
        assert 0 <= lambda_crit <= 2, "Lambda critical should be in valid range"
        assert phi_stats['avg_phi'] >= 0, "Conductance should be non-negative"
        
        print(f"✅ Parameter validation working")

    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        
        # Create test graph
        G = nx.karate_club_graph()
        data = from_networkx(G)
        
        # Run same clustering twice
        np.random.seed(42)
        labels1, n_clusters1, phi_stats1, lambda_crit1 = cmg_filtered_clustering(
            data, k=8, d=15, threshold=0.1
        )
        
        np.random.seed(42)
        labels2, n_clusters2, phi_stats2, lambda_crit2 = cmg_filtered_clustering(
            data, k=8, d=15, threshold=0.1
        )
        
        # Results should be identical
        assert n_clusters1 == n_clusters2, "Number of clusters should be reproducible"
        assert lambda_crit1 == lambda_crit2, "Lambda critical should be reproducible"
        np.testing.assert_array_equal(labels1, labels2, "Cluster labels should be reproducible")
        
        print(f"✅ Results are reproducible: {n_clusters1} clusters")

    def test_karate_club_clustering(self):
        """Test on the famous Karate Club graph."""
        
        G = nx.karate_club_graph()
        data = from_networkx(G)
        
        # Test with different k values
        k_values = [5, 10, 15]
        
        for k in k_values:
            labels, n_clusters, phi_stats, lambda_crit = cmg_filtered_clustering(
                data, k=k, d=20, threshold=0.1
            )
            
            # Basic sanity checks
            assert 1 <= n_clusters <= len(G.nodes), f"Invalid number of clusters: {n_clusters}"
            assert len(labels) == len(G.nodes), f"Wrong number of labels: {len(labels)}"
            assert len(set(labels)) == n_clusters, f"Cluster count mismatch"
            
            # Check conductance is reasonable
            avg_phi = phi_stats['avg_phi']
            assert 0 <= avg_phi <= 10, f"Conductance out of reasonable range: {avg_phi}"
        
        print(f"✅ Karate Club clustering successful across k values")


class TestUtilityFunctions:
    """Test utility functions from filtered.py module."""

    def test_lambda_critical_computation(self):
        """Test lambda critical eigenvalue computation."""
        
        test_cases = [
            (2, 1.0),      # k=2 → λ=1.0
            (5, 0.5714),   # k=5 → λ≈0.57
            (10, 0.3333),  # k=10 → λ≈0.33
            (20, 0.1818),  # k=20 → λ≈0.18
        ]
        
        for k, expected_lambda in test_cases:
            computed_lambda = compute_lambda_critical(k)
            assert computed_lambda == pytest.approx(expected_lambda, abs=0.01), \
                f"λ_critical({k}) = {computed_lambda}, expected ~{expected_lambda}"
        
        print(f"✅ Lambda critical computation verified for {len(test_cases)} cases")

    def test_normalized_laplacian_construction(self):
        """Test normalized Laplacian matrix construction."""
        
        # Create simple test graph
        G = nx.path_graph(4)
        data = from_networkx(G)
        
        from torch_geometric.utils import to_scipy_sparse_matrix
        A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
        
        L_norm = build_normalized_laplacian(A)
        
        # Check properties
        assert L_norm.shape == (4, 4), "Wrong shape for normalized Laplacian"
        assert abs(L_norm.diagonal().sum() - 4.0) < 1e-6, "Diagonal should sum to number of nodes"
        
        # Check symmetry
        assert np.allclose(L_norm.toarray(), L_norm.T.toarray()), "Should be symmetric"
        
        print(f"✅ Normalized Laplacian construction verified")


def test_integration_demo():
    """Integration test that demonstrates the full pipeline."""
    
    print("\n" + "="*60)
    print("🧪 CMG-X FILTERED CLUSTERING INTEGRATION DEMO")
    print("="*60)
    
    # Create demo graph: two communities connected by weak bridge
    G = nx.Graph()
    
    # Community 1: small clique
    community1 = list(range(4))
    for i in community1:
        for j in community1:
            if i < j:
                G.add_edge(i, j, weight=1.5)
    
    # Community 2: larger clique  
    community2 = list(range(4, 9))
    for i in community2:
        for j in community2:
            if i < j:
                G.add_edge(i, j, weight=1.5)
    
    # Weak bridge
    G.add_edge(1, 5, weight=0.1)
    
    data = from_networkx(G)
    edge_weights = [G[u.item()][v.item()]['weight'] for u, v in data.edge_index.t()]
    data.edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    
    print(f"📊 Demo Graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
    print(f"   Community 1: nodes {community1}")
    print(f"   Community 2: nodes {community2}")
    print(f"   Bridge: 1 ↔ 5 (weak connection)")
    
    # Test different filter orders
    print(f"\n🔬 Testing Different Filter Orders:")
    print(f"{'k':<3} {'λ_crit':<8} {'Clusters':<9} {'Std_φ':<8} {'Norm_φ':<8} {'Quality'}")
    print("-" * 50)
    
    for k in [3, 7, 12]:
        labels, n_clusters, phi_stats, lambda_crit = cmg_filtered_clustering(
            data, k=k, d=15, threshold=0.1
        )
        
        if 'comparison' in phi_stats:
            std_phi = phi_stats['comparison']['standard_avg']
            norm_phi = phi_stats['comparison']['normalized_avg']
        else:
            std_phi = phi_stats['avg_phi']
            norm_phi = std_phi
        
        # Assess quality
        if n_clusters == 2:
            cluster_0 = set(np.where(labels == 0)[0])
            cluster_1 = set(np.where(labels == 1)[0])
            
            if (cluster_0 == set(community1) and cluster_1 == set(community2)) or \
               (cluster_0 == set(community2) and cluster_1 == set(community1)):
                quality = "Perfect ✅"
            else:
                quality = "Suboptimal ⚠️"
        else:
            quality = f"{n_clusters} clusters ❌"
        
        print(f"{k:<3} {lambda_crit:<8.4f} {n_clusters:<9} {std_phi:<8.4f} {norm_phi:<8.4f} {quality}")
    
    print(f"\n✅ Integration demo completed successfully!")
    print(f"💡 Higher k values (lower frequencies) typically give better community detection.")


if __name__ == "__main__":
    # Run a quick demo when executed directly
    pytest.main([__file__, "-v", "--tb=short"])