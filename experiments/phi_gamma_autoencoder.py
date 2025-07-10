import torch
import torch.nn as nn
import torch.nn.functional as F

class PhiGammaPooling(nn.Module):
    def __init__(self, in_channels, num_clusters):
        super().__init__()
        self.num_clusters = num_clusters
        self.linear = nn.Linear(in_channels, num_clusters)

    def forward(self, x, adj):
        """
        Args:
            x: [N, F] node features
            adj: [N, N] adjacency matrix (dense or torch sparse)
        Returns:
            X_pooled: [C, F]
            P: [N, C] assignment matrix
            phi_gamma_loss, phi_tensor, gamma_tensor
        """
        N = x.size(0)

        # Assignment matrix
        assign_logits = self.linear(x)
        P = F.softmax(assign_logits, dim=-1)  # [N, C]

        # Pool node features
        X_pooled = P.T @ x  # [C, F]

        # Convert sparse adjacency to dense if needed
        if adj.is_sparse:
            adj = adj.to_dense()

        deg = adj.sum(dim=1)
        D = torch.diag(deg)
        L = D - adj

        phi_list, gamma_list = [], []
        for c in range(self.num_clusters):
            p_c = P[:, c]
            vol = torch.clamp(p_c @ deg, min=1e-6)
            cut = p_c @ (L @ p_c)
            comp = torch.clamp((1 - p_c) @ deg, min=1e-6)
            phi = cut / torch.minimum(vol, comp)

            boundary = p_c @ (adj @ (1 - p_c))
            gamma = boundary / vol

            phi_list.append(phi)
            gamma_list.append(gamma)

        phi_tensor = torch.stack(phi_list)
        gamma_tensor = torch.stack(gamma_list)
        phi_gamma_loss = torch.sum(1.0 / (phi_tensor * gamma_tensor + 1e-6))

        return X_pooled, P, phi_gamma_loss, phi_tensor, gamma_tensor
