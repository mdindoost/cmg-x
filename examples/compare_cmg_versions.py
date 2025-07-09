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
import torch
import time
from cmgx.filtered import cmg_filtered_clustering, evaluate_phi_conductance

def run_baseline_cmg(data):
    """Run baseline CMG clustering with unweighted structural conductance."""
    print("[BASELINE] Running standard CMG clustering")
    
    # Build adjacency matrix
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
    
    # Handle edge weights if present
    if hasattr(data, 'edge_weight') and data.edge_weight is not None:
        rows, cols = data.edge_index[0].numpy(), data.edge_index[1].numpy()
        weights = data.edge_weight.numpy()
        A = sp.coo_matrix((weights, (rows, cols)), shape=(data.num_nodes, data.num_nodes))
        A = A.tocsr()
    
    # Build Laplacian for CMG
    degrees = np.array(A.sum(axis=1)).flatten()
    L = sp.diags(degrees) - A
    
    # Run CMG clustering
    cI_raw, nc = cmgCluster(L.tocsc())
    cI = cI_raw - 1  # Convert to 0-indexed
    
    print(f"[BASELINE] CMG found {nc} clusters")
    print(f"[BASELINE] Raw CMG output (1-indexed): {cI_raw}")
    print(f"[BASELINE] Converted clusters (0-indexed): {cI}")
    print(f"[BASELINE] Cluster sizes: {dict(Counter(cI))}")
    
    # Calculate UNWEIGHTED conductance based on original structure
    phi = evaluate_phi_conductance(data, cI, method='unweighted')

    return cI, nc, phi, A

def print_cluster_analysis(cI, method_name="Method"):
    """Print detailed cluster analysis."""
    counts = Counter(cI)
    total_nodes = len(cI)
    
    print(f"[{method_name}] Cluster Analysis:")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Number of clusters: {len(counts)}")
    
    for cluster_id, size in sorted(counts.items()):
        percentage = 100 * size / total_nodes
        print(f"  Cluster {cluster_id}: {size} nodes ({percentage:.1f}%)")

def print_conductance_details(phi_stats, method_name="Method"):
    """Print detailed unweighted conductance information."""
    print(f"[{method_name}] Unweighted Structural Conductance:")
    
    if 'comparison' in phi_stats:
        print(f"  Standard φ (avg): {phi_stats['comparison']['standard_avg']:.4f}")
        print(f"  Normalized φ (avg): {phi_stats['comparison']['normalized_avg']:.4f}")
        print(f"  Ratio (norm/std): {phi_stats['comparison']['ratio']:.4f}")
        
        print(f"  Per-cluster Standard φ:")
        for cid, phi in sorted(phi_stats['phi'].items()):
            phi_str = f"{phi:.4f}" if phi != float('inf') else "inf"
            print(f"    Cluster {cid}: φ = {phi_str}")
            
        print(f"  Per-cluster Normalized φ:")
        for cid, phi in sorted(phi_stats['normalized_phi'].items()):
            phi_str = f"{phi:.4f}" if phi != float('inf') else "inf"
            print(f"    Cluster {cid}: φ_norm = {phi_str}")
    else:
        print(f"  Average φ: {phi_stats['avg_phi']:.4f}")
        for cid, phi in sorted(phi_stats['phi'].items()):
            phi_str = f"{phi:.4f}" if phi != float('inf') else "inf"
            print(f"    Cluster {cid}: φ = {phi_str}")

def show_cut_edges(data, labels, method_name="Method"):
    """Show which edges are cut by the clustering."""
    print(f"[{method_name}] Cut Edges Analysis:")
    
    edge_index = data.edge_index.cpu().numpy()
    cut_edges = []
    
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        if labels[u] != labels[v]:
            if hasattr(data, 'edge_weight') and data.edge_weight is not None:
                weight = data.edge_weight[i].item()
                cut_edges.append((u, v, weight))
            else:
                cut_edges.append((u, v, 1.0))
    
    print(f"  Total cut edges: {len(cut_edges)}")
    for u, v, w in cut_edges:
        print(f"    {u} ↔ {v} (original weight: {w:.3f})")
    
    return cut_edges

def compare_on_graph(name, data, save_files=True):
    """Compare baseline and filtered CMG using unweighted structural conductance."""
    print(f"\n{'='*60}")
    print(f"TESTING: {name}")
    print(f"{'='*60}")
    print(f"Graph info: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
    
    # Setup file paths
    if save_files:
        os.makedirs("logs", exist_ok=True)
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')
        csv_path = f"logs/{safe_name}_summary.csv"
    
    # === BASELINE CMG ===
    print(f"\n{'-'*30} BASELINE CMG {'-'*30}")
    cI_base, nc_base, phi_base, A_base = run_baseline_cmg(data)
    
    print_cluster_analysis(cI_base, "BASELINE")
    print_conductance_details(phi_base, "BASELINE")
    show_cut_edges(data, cI_base, "BASELINE")
    
    # === FILTERED CMG ===
    print(f"\n{'-'*30} FILTERED CMG {'-'*30}")
    
    filter_configs = [
        {'k': 5, 'd': 20, 'threshold': 0.1, 'name': 'Filtered_k5'},
        {'k': 10, 'd': 20, 'threshold': 0.1, 'name': 'Filtered_k10'},
        {'k': 15, 'd': 20, 'threshold': 0.1, 'name': 'Filtered_k15'},
    ]
    
    results = {'baseline': {
        'clusters': cI_base,
        'num_clusters': nc_base,
        'phi_stats': phi_base
    }}
    
    for config in filter_configs:
        method_name = config.pop('name')
        print(f"\n--- {method_name} ---")
        
        try:
            # Run filtered CMG (it already calculates unweighted conductance internally)
            cI_filt, nc_filt, phi_filt_unweighted, lambda_crit = cmg_filtered_clustering(
                data, conductance_method='unweighted', **config
            )
            
            print(f"[{method_name}] λ_critical = {lambda_crit:.4f}")
            print(f"[{method_name}] Found {nc_filt} clusters")
            print(f"[{method_name}] Cluster assignment: {cI_filt}")
            
            print_cluster_analysis(cI_filt, method_name)
            print_conductance_details(phi_filt_unweighted, method_name)
            show_cut_edges(data, cI_filt, method_name)
            
            results[method_name] = {
                'clusters': cI_filt,
                'num_clusters': nc_filt,
                'phi_stats': phi_filt_unweighted,  # Already unweighted from cmg_filtered_clustering
                'lambda_critical': lambda_crit
            }
                
        except Exception as e:
            print(f"[{method_name}] FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # === COMPARISON SUMMARY ===
    print(f"\n{'-'*30} COMPARISON SUMMARY {'-'*30}")
    
    comparison_data = []
    for method, result in results.items():
        phi_stats = result['phi_stats']
        
        if 'comparison' in phi_stats:
            std_phi = phi_stats['comparison']['standard_avg']
            norm_phi = phi_stats['comparison']['normalized_avg']
        else:
            std_phi = phi_stats['avg_phi']
            norm_phi = phi_stats.get('normalized_avg_phi', std_phi)
        
        comparison_data.append({
            'Method': method,
            'Clusters': result['num_clusters'],
            'Std_φ': f"{std_phi:.4f}" if std_phi != float('inf') else "inf",
            'Norm_φ': f"{norm_phi:.4f}" if norm_phi != float('inf') else "inf"
        })
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    if save_files:
        df.to_csv(csv_path, index=False)
        print(f"\n📊 Results saved to {csv_path}")
    
    return results

def generate_two_cliques_with_bridge():
    """Generate two cliques connected by a weak bridge."""
    print("[GRAPH] Creating two cliques with weak bridge")
    
    G = nx.Graph()
    
    # Clique 1: complete graph on nodes 0-4
    for i in range(5):
        for j in range(i+1, 5):
            G.add_edge(i, j, weight=2.0)
    
    # Clique 2: complete graph on nodes 5-9
    for i in range(5, 10):
        for j in range(i+1, 10):
            G.add_edge(i, j, weight=2.0)
    
    # Weak bridge
    G.add_edge(2, 7, weight=0.05)
    
    print(f"[GRAPH] Created graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"[GRAPH] Clique 1: nodes [0, 1, 2, 3, 4]")
    print(f"[GRAPH] Clique 2: nodes [5, 6, 7, 8, 9]")
    print(f"[GRAPH] Bridge: 2 -- 7 (weight: 0.05)")
    
    data = from_networkx(G)
    edge_weights = [G[u.item()][v.item()]['weight'] for u, v in data.edge_index.t()]
    data.edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    
    return data

def test_specific_examples():
    """Test on the two cliques example with unweighted conductance."""
    
    print("🎯 Testing Two Cliques with Bridge (Unweighted Structural Conductance)")
    clique_data = generate_two_cliques_with_bridge()
    results_cliques = compare_on_graph("Two Cliques with Bridge", clique_data)
    
    # Analyze results for structural clustering quality
    print(f"\n🔍 Analysis for Two Cliques (Unweighted Structural):")
    for method, result in results_cliques.items():
        clusters = result['clusters']
        nc = result['num_clusters']
        phi_stats = result['phi_stats']
        
        if nc == 2:
            # Check if clusters correspond to cliques
            cluster_0_nodes = set(np.where(clusters == 0)[0])
            cluster_1_nodes = set(np.where(clusters == 1)[0])
            
            clique1_set = set(range(5))
            clique2_set = set(range(5, 10))
            
            if ((cluster_0_nodes == clique1_set and cluster_1_nodes == clique2_set) or 
                (cluster_0_nodes == clique2_set and cluster_1_nodes == clique1_set)):
                
                # Perfect structural clustering
                if 'comparison' in phi_stats:
                    std_phi = phi_stats['comparison']['standard_avg']
                    norm_phi = phi_stats['comparison']['normalized_avg']
                    print(f"  ✅ {method}: Perfect clustering! "
                          f"φ_std={std_phi:.4f}, φ_norm={norm_phi:.4f}")
                else:
                    print(f"  ✅ {method}: Perfect clustering! φ={phi_stats['avg_phi']:.4f}")
                    
            else:
                print(f"  ⚠️  {method}: Suboptimal clustering")
                print(f"      Cluster 0: {sorted(cluster_0_nodes)}")
                print(f"      Cluster 1: {sorted(cluster_1_nodes)}")
        else:
            print(f"  ❌ {method}: Found {nc} clusters, expected 2")
    
    # Show the expected vs actual cut
    print(f"\n📊 Expected Cut Analysis:")
    print(f"  Expected cut: 1 edge (2 ↔ 7)")
    print(f"  Perfect clustering should have:")
    print(f"    - Standard φ = 1 / min(21, 21) = 0.0476")  # 1 cut edge, 21 degree each clique
    print(f"    - Normalized φ = 1 / 5 = 0.2000")  # 1 cut edge, 5 nodes per cluster
    
    return results_cliques

def analyze_conductance_theory():
    """Analyze the theoretical conductance values for the two cliques."""
    
    print(f"\n📚 Theoretical Conductance Analysis for Two Cliques")
    print(f"{'='*55}")
    
    print(f"Graph structure:")
    print(f"  - Clique 1: 5 nodes, 10 internal edges")
    print(f"  - Clique 2: 5 nodes, 10 internal edges") 
    print(f"  - Bridge: 1 edge (2 ↔ 7)")
    print(f"  - Total: 21 edges")
    
    print(f"\nUnweighted degree calculation:")
    print(f"  - Each clique node degree = 4 (internal) + 0 or 1 (bridge)")
    print(f"  - Node 2 degree = 4 + 1 = 5")
    print(f"  - Node 7 degree = 4 + 1 = 5") 
    print(f"  - All other nodes degree = 4")
    print(f"  - Total degree = 8*4 + 2*5 = 42")
    
    print(f"\nPerfect clustering (clique 1 = cluster 0, clique 2 = cluster 1):")
    print(f"  - Cut edges: 1 (the bridge 2 ↔ 7)")
    print(f"  - Degree of cluster 0: 4+5+4+4+4 = 21")
    print(f"  - Degree of cluster 1: 5+4+4+4+4 = 21")
    print(f"  - Standard φ = cut / min(deg_0, deg_1) = 1 / min(21, 21) = 1/21 ≈ 0.0476")
    print(f"  - Normalized φ = cut / |cluster| = 1 / 5 = 0.2000")
    
    print(f"\nWorst clustering (each node in separate cluster):")
    print(f"  - Cut edges: 21 (all edges)")
    print(f"  - Each singleton cluster has degree = node degree")
    print(f"  - Standard φ = very high (close to 1.0)")
    print(f"  - Normalized φ = degree / 1 = node degree")

# ============================================================================
# COMPREHENSIVE TESTING FUNCTIONS
# ============================================================================

def run_all_graph_types():
    """Test all requested graph types and create comparison table."""
    
    print("🧪 Testing All Graph Types for CMG Comparison")
    print("=" * 60)
    
    # Create all test graphs
    graphs = {}
    
    print("📋 Creating test graphs...")
    
    # Two cliques (reference)
    graphs['Two Cliques'] = generate_two_cliques_with_bridge()
    
    # # Grid graphs
    # print("  Grid 5x5...")
    # G = nx.grid_2d_graph(5, 5)
    # mapping = {(i, j): i * 5 + j for i in range(5) for j in range(5)}
    # G = nx.relabel_nodes(G, mapping)
    # graphs['Grid 5x5'] = from_networkx(G)
    
    # print("  Grid 10x10...")
    # G = nx.grid_2d_graph(10, 10)
    # mapping = {(i, j): i * 10 + j for i in range(10) for j in range(10)}
    # G = nx.relabel_nodes(G, mapping)
    # graphs['Grid 10x10'] = from_networkx(G)
    
    # Path graphs
    print("  Path 10...")
    graphs['Path 10'] = from_networkx(nx.path_graph(10))
    
    print("  Path 100...")
    graphs['Path 100'] = from_networkx(nx.path_graph(100))
    
    # # ER graphs
    # print("  ER (50, 0.1)...")
    # G = nx.erdos_renyi_graph(50, 0.1, seed=42)
    # G.remove_nodes_from(list(nx.isolates(G)))
    # G = nx.convert_node_labels_to_integers(G)
    # graphs['ER (50, 0.1)'] = from_networkx(G)
    
    # print("  ER (1000, 0.05)...")
    # G = nx.erdos_renyi_graph(1000, 0.05, seed=42)
    # G.remove_nodes_from(list(nx.isolates(G)))
    # G = nx.convert_node_labels_to_integers(G)
    # graphs['ER (1000, 0.05)'] = from_networkx(G)
    
    # # Small-world graph (n=200, k=6, p=0.3)
    # print("  Small-world (200)...")
    # G = nx.watts_strogatz_graph(200, 6, 0.3, seed=42)
    # graphs['Small-World (200)'] = from_networkx(G)
    
    # # Scale-free graph (n=200, m=3)
    # print("  Scale-free (200)...")
    # G = nx.barabasi_albert_graph(200, 3, seed=42)
    # graphs['Scale-Free (200)'] = from_networkx(G)
    
    print(f"✅ Created {len(graphs)} test graphs")
    
    # Print graph statistics
    print(f"\n📊 Graph Statistics:")
    for name, data in graphs.items():
        print(f"  {name:20}: {data.num_nodes:4} nodes, {data.edge_index.shape[1]:5} edges")
    
    # Define test methods
    methods = {
        'baseline': None,
        'k5': {'k': 5, 'd': 20, 'threshold': 0.1},
        'k10': {'k': 10, 'd': 20, 'threshold': 0.1},
        'k15': {'k': 15, 'd': 20, 'threshold': 0.1},
    }
    
    # Run tests and collect results
    all_results = []
    
    for graph_name, data in graphs.items():
        print(f"\n--- Testing {graph_name} ---")
        
        for method_name, config in methods.items():
            print(f"  {method_name}...", end=' ')
            
            start_time = time.time()
            
            try:
                if method_name == 'baseline':
                    # Baseline CMG
                    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
                    
                    if hasattr(data, 'edge_weight') and data.edge_weight is not None:
                        rows, cols = data.edge_index[0].numpy(), data.edge_index[1].numpy()
                        weights = data.edge_weight.numpy()
                        A = sp.coo_matrix((weights, (rows, cols)), shape=(data.num_nodes, data.num_nodes))
                        A = A.tocsr()
                    
                    degrees = np.array(A.sum(axis=1)).flatten()
                    L = sp.diags(degrees) - A
                    
                    cI_raw, nc = cmgCluster(L.tocsc())
                    cI = cI_raw - 1
                    
                    phi_stats = evaluate_phi_conductance(data, cI, method='unweighted')
                    
                else:
                    # Filtered CMG
                    cI, nc, phi_stats, lambda_crit = cmg_filtered_clustering(
                        data, conductance_method='unweighted', **config
                    )
                
                elapsed_time = time.time() - start_time
                
                # Extract conductance values
                if 'comparison' in phi_stats:
                    std_phi = phi_stats['comparison']['standard_avg']
                    norm_phi = phi_stats['comparison']['normalized_avg']
                else:
                    std_phi = phi_stats['avg_phi']
                    norm_phi = phi_stats.get('normalized_avg_phi', std_phi)
                
                all_results.append({
                    'Graph': graph_name,
                    'Method': method_name,
                    'Clusters': nc,
                    'Std_φ': f"{std_phi:.4f}" if std_phi != float('inf') else "inf",
                    'Norm_φ': f"{norm_phi:.4f}" if norm_phi != float('inf') else "inf",
                    'Time(s)': f"{elapsed_time:.3f}",
                    'Status': '✅'
                })
                
                print(f"✅ {nc} clusters, φ_std={std_phi:.4f}")
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                all_results.append({
                    'Graph': graph_name,
                    'Method': method_name,
                    'Clusters': 'FAIL',
                    'Std_φ': 'FAIL',
                    'Norm_φ': 'FAIL',
                    'Time(s)': f"{elapsed_time:.3f}",
                    'Status': '❌'
                })
                print(f"❌ FAILED: {str(e)[:40]}...")
    
    # Create comprehensive results table
    df = pd.DataFrame(all_results)
    
    print(f"\n{'='*100}")
    print("COMPREHENSIVE RESULTS TABLE")
    print(f"{'='*100}")
    print(df.to_string(index=False))
    
    # Save results
    os.makedirs("logs", exist_ok=True)
    df.to_csv("logs/comprehensive_graph_comparison.csv", index=False)
    print(f"\n📊 Results saved to logs/comprehensive_graph_comparison.csv")
    
    # Analysis: Best method per graph
    print(f"\n🏆 BEST METHOD PER GRAPH (by Standard Conductance):")
    print("-" * 60)
    
    successful_results = df[df['Status'] == '✅'].copy()
    
    for graph in df['Graph'].unique():
        graph_data = successful_results[successful_results['Graph'] == graph].copy()  # Add .copy() here
        if len(graph_data) > 0:
            # Convert conductance to numeric for comparison
            graph_data.loc[:, 'std_phi_numeric'] = pd.to_numeric(graph_data['Std_φ'], errors='coerce')  # Use .loc
            
            # Find best method (lowest conductance)
            best_idx = graph_data['std_phi_numeric'].idxmin()
            best_row = graph_data.loc[best_idx]
            
            print(f"  {graph:20} → {best_row['Method']:10} "
                  f"(φ_std={best_row['Std_φ']:8}, φ_norm={best_row['Norm_φ']:8}, "
                  f"{best_row['Clusters']:3} clusters)")
    
    # Method performance summary
    print(f"\n📈 METHOD PERFORMANCE SUMMARY:")
    print("-" * 60)
    
    if len(successful_results) > 0:
        method_summary = []
        for method in methods.keys():
            method_data = successful_results[successful_results['Method'] == method]
            if len(method_data) > 0:
                avg_std_phi = pd.to_numeric(method_data['Std_φ'], errors='coerce').mean()
                avg_norm_phi = pd.to_numeric(method_data['Norm_φ'], errors='coerce').mean()
                avg_time = pd.to_numeric(method_data['Time(s)'], errors='coerce').mean()
                success_rate = len(method_data) / len(df[df['Method'] == method]) * 100
                
                method_summary.append({
                    'Method': method,
                    'Avg_Std_φ': f"{avg_std_phi:.4f}",
                    'Avg_Norm_φ': f"{avg_norm_phi:.4f}",
                    'Avg_Time(s)': f"{avg_time:.3f}",
                    'Success_Rate': f"{success_rate:.1f}%"
                })
        
        summary_df = pd.DataFrame(method_summary)
        print(summary_df.to_string(index=False))
    
    # Graph complexity analysis
    print(f"\n🔍 GRAPH COMPLEXITY ANALYSIS:")
    print("-" * 60)
    
    graph_complexity = []
    for graph in df['Graph'].unique():
        graph_data = successful_results[successful_results['Graph'] == graph]
        if len(graph_data) > 0:
            # Find the graph info
            data = graphs[graph]
            density = data.edge_index.shape[1] / (data.num_nodes * (data.num_nodes - 1) / 2)
            
            # Average conductance across all methods
            avg_conductance = pd.to_numeric(graph_data['Std_φ'], errors='coerce').mean()
            
            graph_complexity.append({
                'Graph': graph,
                'Nodes': data.num_nodes,
                'Edges': data.edge_index.shape[1],
                'Density': f"{density:.4f}",
                'Avg_φ': f"{avg_conductance:.4f}",
                'Complexity': 'High' if avg_conductance > 0.5 else 'Medium' if avg_conductance > 0.1 else 'Low'
            })
    
    complexity_df = pd.DataFrame(graph_complexity)
    print(complexity_df.to_string(index=False))
    
    return df

def main():
    """Updated main function with comprehensive tests option."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CMG Clustering Comparison')
    parser.add_argument('--mode', choices=['cliques', 'theory', 'comprehensive'], 
                       default='comprehensive', help='Test mode')
    
    args = parser.parse_args()
    
    if args.mode == 'cliques':
        print("🚀 Running CMG Comparison with Unweighted Structural Conductance")
        analyze_conductance_theory()
        test_specific_examples()
    elif args.mode == 'theory':
        analyze_conductance_theory()
    elif args.mode == 'comprehensive':
        run_all_graph_types()

if __name__ == '__main__':
    # For direct execution, run the comprehensive test
    print("🚀 Running Comprehensive CMG Analysis on All Graph Types")
    print("This will test 9 different graph types with 4 different methods each")
    print("Expected runtime: 3-10 minutes depending on system")
    print()
    
    start_time = time.time()
    
    try:
        results_df = run_all_graph_types()
        
        total_time = time.time() - start_time
        print(f"\n🎉 Comprehensive analysis completed in {total_time:.1f} seconds!")
        print(f"Check logs/comprehensive_graph_comparison.csv for detailed results")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()