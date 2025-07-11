import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from phi_gamma_autoencoder import PhiGammaPooling, unpool


class PhiGammaNodeClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_clusters, num_classes, recon_method='soft'):
        super().__init__()
        self.gcn = GCNConv(in_channels, hidden_channels)
        self.pool = PhiGammaPooling(hidden_channels, num_clusters)
        self.recon_method = recon_method
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, adj, edge_index):
        # Use GCN encoder instead of MLP
        x_enc = F.relu(self.gcn(x, edge_index))  # Graph-aware encoding

        # Pool features using CMG-style cluster assignment
        X_pooled, P, phi_gamma_loss, phi_tensor, gamma_tensor = self.pool(x_enc, adj)

        # Unpool back to node space
        degrees = adj.to_dense().sum(dim=1) if adj.is_sparse else adj.sum(dim=1)
        cluster_ids = P.argmax(dim=1)

        x_unpooled = unpool(P, X_pooled, method=self.recon_method,
                            cluster_assignments=cluster_ids, degrees=degrees)

        logits = self.classifier(x_unpooled)
        return logits, phi_gamma_loss, P, phi_tensor, gamma_tensor
