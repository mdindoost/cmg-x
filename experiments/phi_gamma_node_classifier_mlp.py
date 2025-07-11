import torch
import torch.nn as nn
import torch.nn.functional as F
from phi_gamma_autoencoder import PhiGammaPooling, unpool


class PhiGammaNodeClassifierMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_clusters, num_classes, recon_method='soft'):
        super().__init__()

        # 2-layer MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_channels),
            nn.ReLU()
        )

        self.pool = PhiGammaPooling(hidden_channels, num_clusters)
        self.recon_method = recon_method
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, adj):
        # No edge_index — just MLP
        x_enc = self.encoder(x)

        X_pooled, P, phi_gamma_loss, phi_tensor, gamma_tensor = self.pool(x_enc, adj)

        degrees = adj.to_dense().sum(dim=1) if adj.is_sparse else adj.sum(dim=1)
        cluster_ids = P.argmax(dim=1)

        x_unpooled = unpool(P, X_pooled, method=self.recon_method,
                            cluster_assignments=cluster_ids, degrees=degrees)

        logits = self.classifier(x_unpooled)
        return logits, phi_gamma_loss, P, phi_tensor, gamma_tensor
