"""
CMG-U-Net with Channel Pyramid Architecture
==========================================

Implements GraphU-Net style channel pyramid:
- Encoder: Increase channels as graph gets smaller  
- Decoder: Decrease channels as graph gets larger
- CMG pooling for principled graph coarsening
- Skip connections with channel alignment

Architecture:
Level 0: N nodes, 64 channels → CMG Pool → Level 1: k₁ nodes, 128 channels
Level 1: k₁ nodes, 128 channels → CMG Pool → Level 2: k₂ nodes, 256 channels  
Level 2: k₂ nodes, 256 channels → CMG Pool → Level 3: k₃ nodes, 512 channels (bottleneck)
Level 3: k₃ nodes, 512 channels → CMG Unpool → Level 2: k₂ nodes, 256 channels (+ skip)
Level 2: k₂ nodes, 256+256 channels → CMG Unpool → Level 1: k₁ nodes, 128 channels (+ skip) 
Level 1: k₁ nodes, 128+128 channels → CMG Unpool → Level 0: N nodes, 64 channels (+ skip)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from typing import List, Tuple, Dict
import numpy as np

# Import pooling components
try:
    from phi_gamma_autoencoder_fixed import PhiGammaPooling, unpool

except ImportError:
    print("⚠️  Warning: Could not import from phi_gamma_autoencoder")
    print("   Make sure phi_gamma_autoencoder.py is in the same directory")
    raise


def adj_to_edge_index(adj_matrix: torch.Tensor) -> torch.Tensor:
    """Convert adjacency matrix to edge_index format."""
    adj_matrix = adj_matrix.float()
    
    # Handle edge case of very small graphs
    if adj_matrix.shape[0] <= 1:
        # Return empty edge index for single node or empty graphs
        return torch.empty((2, 0), dtype=torch.long, device=adj_matrix.device)
    
    # Remove self-loops and very small weights
    adj_matrix.fill_diagonal_(0)
    adj_matrix = torch.where(adj_matrix > 1e-6, adj_matrix, 0)
    
    edge_index = adj_matrix.nonzero().t().contiguous()
    
    # If no edges, create a minimal complete graph for very small graphs
    if edge_index.shape[1] == 0 and adj_matrix.shape[0] <= 5:
        nodes = torch.arange(adj_matrix.shape[0], device=adj_matrix.device)
        edge_index = torch.combinations(nodes, r=2).t().contiguous()
        if edge_index.shape[1] > 0:
            # Add reverse edges
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    return edge_index


class CMGPyramidLevel(nn.Module):
    """
    Single level of CMG-U-Net with channel pyramid.
    Handles both encoding (pooling) and decoding (unpooling).
    """
    
    def __init__(self, 
                 encoder_in_channels: int,
                 encoder_out_channels: int, 
                 decoder_in_channels: int,
                 decoder_out_channels: int,
                 num_clusters: int,
                 gnn_type: str = 'gcn',
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_clusters = num_clusters
        self.gnn_type = gnn_type
        
        # Encoder: increase channels while pooling
        if gnn_type == 'gcn':
            self.encoder_gnn = GCNConv(encoder_in_channels, encoder_out_channels)
        elif gnn_type == 'gat':
            self.encoder_gnn = GATConv(encoder_in_channels, encoder_out_channels, heads=1, concat=False)
        
        # CMG pooling
        self.pooling = PhiGammaPooling(encoder_out_channels, num_clusters)
        
        # Decoder: combine skip + unpooled features, then reduce channels
        if gnn_type == 'gcn':
            self.decoder_gnn = GCNConv(decoder_in_channels, decoder_out_channels)
        elif gnn_type == 'gat':
            self.decoder_gnn = GATConv(decoder_in_channels, decoder_out_channels, heads=1, concat=False)
        
        # Channel alignment for skip connections
        # Skip features have encoder_out_channels, decoder expects decoder_out_channels for skip
        self.skip_projection = nn.Linear(encoder_out_channels, encoder_out_channels)  # Keep same size
        
        self.dropout = nn.Dropout(dropout)
        self.encoder_norm = nn.LayerNorm(encoder_out_channels)
        self.decoder_norm = nn.LayerNorm(decoder_out_channels)
        
    def encode(self, x: torch.Tensor, adj: torch.Tensor, edge_index: torch.Tensor) -> Dict:
        """Encoding pass: GNN + CMG pooling + channel increase."""
        # Apply encoder GNN (increase channels)
        x_encoded = F.relu(self.encoder_gnn(x, edge_index))
        x_encoded = self.encoder_norm(x_encoded)
        x_encoded = self.dropout(x_encoded)
        
        # Apply CMG pooling
        x_pooled, P, phi_gamma_loss, phi_tensor, gamma_tensor = self.pooling(x_encoded, adj)
        
        # Create coarse adjacency matrix using your approach
        adj_coarse = P.T @ adj @ P
        
        return {
            'x_pooled': x_pooled,
            'adj_coarse': adj_coarse,
            'assignment_matrix': P,
            'phi_gamma_loss': phi_gamma_loss,
            'phi_tensor': phi_tensor,
            'gamma_tensor': gamma_tensor,
            'skip_features': x_encoded  # Store for skip connection
        }
    
    def decode(self, x_coarse: torch.Tensor, assignment_matrix: torch.Tensor, 
               skip_features: torch.Tensor, original_adj: torch.Tensor, 
               edge_index: torch.Tensor) -> torch.Tensor:
        """Decoding pass: Unpool + skip connection + GNN + channel decrease."""
        # Unpool from coarse to fine resolution
        degrees = original_adj.sum(dim=1)
        cluster_assignments = assignment_matrix.argmax(dim=1)
        x_unpooled = unpool(assignment_matrix, x_coarse, method='soft',
                           cluster_assignments=cluster_assignments, degrees=degrees)
        
        # Project skip features to match decoder output channels if needed
        skip_projected = self.skip_projection(skip_features)
        
        # Combine unpooled features with skip connection
        x_combined = torch.cat([x_unpooled, skip_projected], dim=1)
        
        # Apply decoder GNN (decrease channels)
        x_decoded = F.relu(self.decoder_gnn(x_combined, edge_index))
        x_decoded = self.decoder_norm(x_decoded)
        x_decoded = self.dropout(x_decoded)
        
        return x_decoded


class CMGUNetPyramid(nn.Module):
    """
    CMG-U-Net with GraphU-Net style channel pyramid.
    """
    
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 channels: List[int] = [64, 128, 256, 512],
                 estimated_clusters: List[int] = None,
                 gnn_type: str = 'gcn',
                 dropout: float = 0.1,
                 phi_gamma_weight: float = 1e-3):
        super().__init__()
        
        self.num_levels = len(channels) - 1  # Number of pooling operations
        self.channels = channels
        self.phi_gamma_weight = phi_gamma_weight
        
        # Default cluster estimates if not provided
        if estimated_clusters is None:
            estimated_clusters = [50, 20, 5]  # Rough defaults
        self.estimated_clusters = estimated_clusters
        
        print(f"CMG-U-Net Pyramid Architecture:")
        print(f"  Channels: {channels}")
        print(f"  Estimated clusters: {estimated_clusters}")
        print(f"  Levels: {self.num_levels}")
        
        # Input projection to first channel size
        self.input_proj = nn.Linear(input_dim, channels[0])
        
        # Encoder levels (contracting path)
        self.encoder_levels = nn.ModuleList()
        for i in range(self.num_levels):
            # For decoder: unpooled from next level + skip from current level encoder
            # Unpooled: channels[i+1], Skip: channels[i+1] (encoder output)
            decoder_input_channels = channels[i + 1] + channels[i + 1]
            
            level = CMGPyramidLevel(
                encoder_in_channels=channels[i],
                encoder_out_channels=channels[i + 1], 
                decoder_in_channels=decoder_input_channels,
                decoder_out_channels=channels[i],
                num_clusters=estimated_clusters[i] if i < len(estimated_clusters) else 5,
                gnn_type=gnn_type,
                dropout=dropout
            )
            self.encoder_levels.append(level)
        
        # Bottleneck layer (no GNN, just feature transformation)
        self.bottleneck = nn.Sequential(
            nn.Linear(channels[-1], channels[-1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.bottleneck_norm = nn.LayerNorm(channels[-1])
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[0], channels[0] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels[0] // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor, edge_index: torch.Tensor) -> Dict:
        """Forward pass through pyramid architecture."""
        # Input projection
        x = F.relu(self.input_proj(x))
        x = self.dropout(x)
        
        # Storage for skip connections and intermediate results
        skip_connections = []
        assignment_matrices = []
        adjacency_matrices = [adj]
        edge_indices = [edge_index]
        total_phi_gamma_loss = 0.0
        
        # Encoder path (contracting - increase channels, decrease nodes)
        current_x = x
        current_adj = adj
        current_edge_index = edge_index
        
        print(f"Encoder path:")
        for i, encoder_level in enumerate(self.encoder_levels):
            print(f"  Level {i}: {current_x.shape[0]} nodes, {current_x.shape[1]} channels → ", end="")
            
            # Encode at this level
            encoded = encoder_level.encode(current_x, current_adj, current_edge_index)
            
            # Store for skip connections and decoding
            skip_connections.append(encoded['skip_features'])
            assignment_matrices.append(encoded['assignment_matrix'])
            adjacency_matrices.append(encoded['adj_coarse'])
            total_phi_gamma_loss += encoded['phi_gamma_loss']
            
            # Update for next level
            current_x = encoded['x_pooled']
            current_adj = encoded['adj_coarse']
            current_edge_index = adj_to_edge_index(current_adj)
            edge_indices.append(current_edge_index)
            
            print(f"{current_x.shape[0]} nodes, {current_x.shape[1]} channels")
        
        # Bottleneck (no edge operations needed - just feature processing)
        print(f"  Bottleneck: {current_x.shape[0]} nodes, {current_x.shape[1]} channels")
        x_bottleneck = self.bottleneck(current_x)
        x_bottleneck = self.bottleneck_norm(x_bottleneck)
        
        # Decoder path (expanding - decrease channels, increase nodes)
        current_x = x_bottleneck
        
        print(f"Decoder path:")
        for i in range(self.num_levels):
            # Get corresponding encoder level (reverse order)
            encoder_idx = self.num_levels - 1 - i
            decoder_level = self.encoder_levels[encoder_idx]
            
            print(f"  Level {encoder_idx}: {current_x.shape[0]} nodes, {current_x.shape[1]} channels → ", end="")
            
            # Get stored information for this level
            skip_features = skip_connections[encoder_idx]
            assignment_matrix = assignment_matrices[encoder_idx]
            original_adj = adjacency_matrices[encoder_idx]
            original_edge_index = edge_indices[encoder_idx]
            
            # Decode
            current_x = decoder_level.decode(
                current_x, assignment_matrix, skip_features, 
                original_adj, original_edge_index
            )
            
            print(f"{current_x.shape[0]} nodes, {current_x.shape[1]} channels")
        
        # Final classification
        logits = self.classifier(current_x)
        
        return {
            'logits': logits,
            'phi_gamma_loss': total_phi_gamma_loss,
            'assignment_matrices': assignment_matrices,
            'skip_connections': skip_connections,
            'adjacency_matrices': adjacency_matrices
        }
    
    def get_pyramid_info(self) -> Dict:
        """Get information about the pyramid structure."""
        return {
            'channels': self.channels,
            'estimated_clusters': self.estimated_clusters,
            'num_levels': self.num_levels,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class CMGUNetPyramidTrainer:
    """Training utilities for CMG-U-Net Pyramid."""
    
    def __init__(self, model: CMGUNetPyramid, phi_gamma_weight: float = 1e-3):
        self.model = model
        self.phi_gamma_weight = phi_gamma_weight
    
    def compute_loss(self, outputs: Dict, targets: torch.Tensor, mask: torch.Tensor) -> Dict:
        """Compute total loss including classification and pyramid structural losses."""
        logits = outputs['logits']
        phi_gamma_loss = outputs['phi_gamma_loss']
        
        # Classification loss
        ce_loss = F.cross_entropy(logits[mask], targets[mask])
        
        # Optional: Multi-scale consistency loss
        # Could add losses that ensure predictions are consistent across pyramid levels
        
        # Total loss
        total_loss = ce_loss + self.phi_gamma_weight * phi_gamma_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'phi_gamma_loss': phi_gamma_loss,
            'phi_gamma_weighted': self.phi_gamma_weight * phi_gamma_loss
        }
    
    def evaluate(self, outputs: Dict, targets: torch.Tensor, mask: torch.Tensor) -> float:
        """Evaluate accuracy."""
        with torch.no_grad():
            logits = outputs['logits']
            pred = logits.argmax(dim=1)
            correct = (pred[mask] == targets[mask]).sum().item()
            total = mask.sum().item()
            return correct / total if total > 0 else 0.0
    
    def analyze_pyramid_structure(self, outputs: Dict) -> Dict:
        """Analyze the learned pyramid structure."""
        assignment_matrices = outputs['assignment_matrices']
        
        analysis = {
            'level_sizes': [],
            'cluster_distributions': [],
            'assignment_entropy': []
        }
        
        for i, P in enumerate(assignment_matrices):
            # Cluster sizes at this level
            cluster_sizes = P.sum(dim=0).detach().cpu().numpy()
            analysis['level_sizes'].append(len(cluster_sizes))
            analysis['cluster_distributions'].append(cluster_sizes)
            
            # Assignment entropy (how "hard" vs "soft" are the assignments)
            entropy = -(P * torch.log(P + 1e-8)).sum(dim=1).mean().detach().item()
            analysis['assignment_entropy'].append(entropy)
        
        return analysis


def create_cmg_unet_pyramid_config(dataset_name: str, estimated_clusters: int) -> Dict:
    """Create pyramid configuration based on dataset characteristics."""
    
    configs = {
        'Cora': {
            'channels': [64, 128, 256, 512],
            'estimated_clusters': [estimated_clusters // 2, estimated_clusters // 4, 3],
            'dropout': 0.2,
            'phi_gamma_weight': 1e-3,
            'gnn_type': 'gcn'
        },
        'CiteSeer': {
            'channels': [64, 128, 256, 512],
            'estimated_clusters': [estimated_clusters // 2, estimated_clusters // 4, 3],
            'dropout': 0.3,
            'phi_gamma_weight': 1e-3,
            'gnn_type': 'gcn'
        },
        'PubMed': {
            'channels': [128, 256, 512, 1024],
            'estimated_clusters': [estimated_clusters // 2, estimated_clusters // 4, estimated_clusters // 8],
            'dropout': 0.1,
            'phi_gamma_weight': 5e-4,
            'gnn_type': 'gat'
        }
    }
    
    return configs.get(dataset_name, configs['Cora'])


# Test function
def test_pyramid_architecture():
    """Test the pyramid architecture."""
    print("Testing CMG-U-Net Pyramid Architecture")
    print("=" * 50)
    
    # Create dummy data
    num_nodes = 100
    input_dim = 16
    num_classes = 7
    
    x = torch.randn(num_nodes, input_dim)
    adj = torch.rand(num_nodes, num_nodes)
    adj = (adj + adj.T) / 2  # Make symmetric
    adj.fill_diagonal_(0)   # Remove self-loops
    edge_index = adj.nonzero().t().contiguous()
    
    # Create model
    model = CMGUNetPyramid(
        input_dim=input_dim,
        num_classes=num_classes,
        channels=[32, 64, 128, 256],
        estimated_clusters=[50, 20, 5],
        gnn_type='gcn',
        dropout=0.1
    )
    
    # Print model info
    pyramid_info = model.get_pyramid_info()
    print(f"Model parameters: {pyramid_info['total_parameters']:,}")
    print(f"Trainable parameters: {pyramid_info['trainable_parameters']:,}")
    
    # Forward pass
    try:
        outputs = model(x, adj, edge_index)
        print(f"\n✅ Forward pass successful!")
        print(f"Output logits shape: {outputs['logits'].shape}")
        print(f"Phi-gamma loss: {outputs['phi_gamma_loss'].item():.4f}")
        
        # Test trainer
        trainer = CMGUNetPyramidTrainer(model)
        dummy_targets = torch.randint(0, num_classes, (num_nodes,))
        dummy_mask = torch.ones(num_nodes, dtype=torch.bool)
        
        loss_dict = trainer.compute_loss(outputs, dummy_targets, dummy_mask)
        accuracy = trainer.evaluate(outputs, dummy_targets, dummy_mask)
        
        print(f"Classification loss: {loss_dict['ce_loss'].item():.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Analyze structure
        structure_analysis = trainer.analyze_pyramid_structure(outputs)
        print(f"Pyramid structure: {structure_analysis['level_sizes']}")
        
        return model, outputs
        
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        return None, None


if __name__ == "__main__":
    test_pyramid_architecture()