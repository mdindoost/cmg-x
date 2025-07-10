import torch
import torch.nn as nn
import torch.nn.functional as F

class PhiGammaPooling(nn.Module):
    def __init__(self, in_channels, num_clusters):
        super().__init__()
        self.num_clusters = num_clusters
        self.linear = nn.Linear(in_channels, num_clusters)  # Learn cluster assignment

    def forward(self, x, adj):
        """
        x: Node features [N, F]
        adj: Adjacency matrix [N, N] (dense or sparse tensor)
        """
        N = x.size(0)
        assign_logits = self.linear(x)  # [N, C]
        P = F.softmax(assign_logits, dim=-1)  # [N, C]

        # Compute pooled features
        X_pooled = torch.matmul(P.t(), x)  # [C, F]

        # Build soft Laplacian components
        deg = torch.sum(adj, dim=1)
        D = torch.diag(deg)
        L = D - adj

        phi_list = []
        gamma_list = []

        for c in range(self.num_clusters):
            p_c = P[:, c]  # [N]
            one_vec = torch.ones_like(p_c)

            vol = torch.clamp(torch.matmul(p_c.T, deg), min=1e-6)
            cut = torch.matmul(p_c.T, torch.matmul(L, p_c))
            comp = torch.clamp(torch.matmul((1 - p_c).T, deg), min=1e-6)

            phi = cut / torch.minimum(vol, comp)
            gamma = torch.matmul(p_c.T, torch.matmul(adj, 1 - p_c)) / vol

            phi_list.append(phi)
            gamma_list.append(gamma)

        phi_tensor = torch.stack(phi_list)
        gamma_tensor = torch.stack(gamma_list)
        phi_gamma_loss = torch.sum(1.0 / (phi_tensor * gamma_tensor + 1e-6))  # minimize 1 / (phi * gamma)

        return X_pooled, P, phi_gamma_loss, phi_tensor, gamma_tensor



class GCNWithPhiGamma(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_clusters):
        super().__init__()
        self.conv1 = nn.Linear(in_channels, hidden_channels)
        self.pool = PhiGammaPooling(hidden_channels, num_clusters)
        self.conv2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, adj):
        x = F.relu(self.conv1(x))
        x_pooled, P, phi_gamma_loss, phi_tensor, gamma_tensor = self.pool(x, adj)

        out = self.conv2(x_pooled)  # [C, out_channels]
        return out, phi_gamma_loss, P, phi_tensor, gamma_tensor


