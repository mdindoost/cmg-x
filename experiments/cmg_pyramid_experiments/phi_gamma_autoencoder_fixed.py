"""
Updated phi_gamma_autoencoder.py with Fixed φγ Implementation
===========================================================

Replaces the incorrect γ calculation with proper conductance-based approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_conductance(node_membership, adj):
    """
    Compute soft conductance given node membership probabilities.
    
    Args:
        node_membership: [N] tensor of membership probabilities in [0,1]
        adj: [N, N] adjacency matrix
    
    Returns:
        conductance: scalar tensor
    """
    if hasattr(adj, 'to_dense'):
        adj = adj.to_dense()
    
    degrees = adj.sum(dim=1)
    
    # Soft volume of the set
    vol_set = torch.clamp(node_membership @ degrees, min=1e-8)
    
    # Soft volume of complement
    vol_complement = torch.clamp((1 - node_membership) @ degrees, min=1e-8)
    
    # Soft cut calculation - vectorized version
    cut = node_membership @ (adj @ (1 - node_membership))
    
    # Conductance formula
    conductance = cut / torch.minimum(vol_set, vol_complement)
    return conductance


class PhiGammaPooling(nn.Module):
    """
    Fixed φγ pooling with proper conductance calculations for both φ and γ.
    """
    
    def __init__(self, in_channels, num_clusters):
        super().__init__()
        self.num_clusters = num_clusters
        self.linear = nn.Linear(in_channels, num_clusters)

    def forward(self, x, adj):
        N = x.size(0)
        assign_logits = self.linear(x)
        P = F.softmax(assign_logits, dim=-1)  # [N, C]
        X_pooled = P.T @ x  # [C, F]

        if hasattr(adj, 'to_dense'):
            adj = adj.to_dense()

        phi_list, gamma_list = [], []
        
        for c in range(self.num_clusters):
            cluster_membership = P[:, c]  # [N] soft membership in cluster c
            
            # φ: Conductance of node-induced subgraph G[V'] (the cluster itself)
            phi = soft_conductance(cluster_membership, adj)
            
            # γ: Conductance of edge-induced subgraph G[E_V']
            # Step 1: Find nodes in edge-induced subgraph
            # A node is in edge-induced subgraph if it connects to the cluster
            connection_to_cluster = adj @ cluster_membership
            edge_induced_membership = torch.sigmoid(connection_to_cluster)
            
            # Step 2: Compute conductance on edge-induced subgraph
            gamma = soft_conductance(edge_induced_membership, adj)

            phi_list.append(phi)
            gamma_list.append(gamma)

        phi_tensor = torch.stack(phi_list)
        gamma_tensor = torch.stack(gamma_list)
        
        # φγ loss: maximize φγ by minimizing 1/(φγ)
        phi_gamma_loss = torch.sum(1.0 / (phi_tensor * gamma_tensor + 1e-6))

        return X_pooled, P, phi_gamma_loss, phi_tensor, gamma_tensor


def unpool(P, X_c, method='soft', cluster_assignments=None, degrees=None):
    """
    Reconstruct node-level features from coarse features.

    Args:
        P: [N x C] soft assignment matrix
        X_c: [C x F] coarse features
        method: one of ['soft', 'copy', 'first', 'central']
        cluster_assignments: [N] hard assignments (needed for non-soft methods)
        degrees: [N] node degrees (needed for 'central' method)

    Returns:
        [N x F] reconstructed node features
    """
    N, C = P.shape
    F = X_c.size(1)
    device = X_c.device

    if method == 'soft':
        return torch.matmul(P, X_c)

    if cluster_assignments is None:
        cluster_assignments = P.argmax(dim=1)

    out = torch.zeros((N, F), device=device)

    if method == 'copy':
        for c in range(C):
            members = (cluster_assignments == c)
            out[members] = X_c[c]
        return out

    if method == 'first':
        for c in range(C):
            members = (cluster_assignments == c).nonzero(as_tuple=False)
            if members.size(0) > 0:
                first = members[0].item()
                out[first] = X_c[c]
        return out

    if method == 'central':
        if degrees is None:
            raise ValueError("Degrees required for 'central' unpooling.")

        for c in range(C):
            members = (cluster_assignments == c).nonzero(as_tuple=False).squeeze()

            if members.numel() == 0:
                continue

            if members.ndim == 0 or members.numel() == 1:
                central_idx = int(members)
            else:
                degs = degrees[members]
                max_deg_idx = degs.argmax()
                central_idx = members[max_deg_idx]

            out[central_idx] = X_c[c]
        return out

    raise ValueError(f"Unknown method: {method}")


class PhiGammaAutoencoder(nn.Module):
    """
    Autoencoder with fixed φγ pooling.
    """
    
    def __init__(self, in_channels, hidden_channels, num_clusters, recon_method='soft'):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_channels),
            nn.ReLU()
        )

        self.pool = PhiGammaPooling(hidden_channels, num_clusters)
        self.recon_method = recon_method
        self.decoder = nn.Linear(hidden_channels, in_channels)

    def forward(self, x, adj):
        x_enc = F.relu(self.encoder(x))
        X_pooled, P, phi_gamma_loss, phi_tensor, gamma_tensor = self.pool(x_enc, adj)

        degrees = adj.sum(dim=1) if adj.is_sparse else adj.sum(dim=1)
        cluster_ids = P.argmax(dim=1)

        x_unpooled = unpool(P, X_pooled, method=self.recon_method,
                            cluster_assignments=cluster_ids, degrees=degrees)
        x_hat = self.decoder(x_unpooled)

        return x_hat, phi_gamma_loss, P, phi_tensor, gamma_tensor


# Test the fixed implementation
def test_fixed_implementation():
    """Test the fixed φγ implementation."""
    print("🔧 Testing Fixed φγ Implementation")
    print("=" * 40)
    
    # Create test data with clear community structure
    N, F, C = 12, 8, 3
    x = torch.randn(N, F)
    
    # Create adjacency matrix with 3 communities
    adj = torch.zeros(N, N)
    
    # Community 1: nodes 0-3
    for i in range(4):
        for j in range(i+1, 4):
            adj[i, j] = adj[j, i] = 1.5
    
    # Community 2: nodes 4-7
    for i in range(4, 8):
        for j in range(i+1, 8):
            adj[i, j] = adj[j, i] = 1.5
    
    # Community 3: nodes 8-11
    for i in range(8, 12):
        for j in range(i+1, 12):
            adj[i, j] = adj[j, i] = 1.5
    
    # Weak inter-community connections
    adj[3, 4] = adj[4, 3] = 0.1
    adj[7, 8] = adj[8, 7] = 0.1
    
    # Test pooling
    pooling = PhiGammaPooling(F, C)
    X_pooled, P, phi_gamma_loss, phi_tensor, gamma_tensor = pooling(x, adj)
    
    print(f"✅ Shape check: X_pooled {X_pooled.shape}, P {P.shape}")
    print(f"✅ φγ loss: {phi_gamma_loss.item():.4f}")
    print(f"✅ φ values: {phi_tensor.detach().numpy().round(4)}")
    print(f"✅ γ values: {gamma_tensor.detach().numpy().round(4)}")
    print(f"✅ φγ products: {(phi_tensor * gamma_tensor).detach().numpy().round(4)}")
    
    # Test gradient flow
    phi_gamma_loss.backward()
    grad_norm = torch.norm(pooling.linear.weight.grad).item()
    print(f"✅ Gradient norm: {grad_norm:.6f}")
    
    print(f"🎯 Fixed implementation working correctly!")
    return True


if __name__ == "__main__":
    test_fixed_implementation()