import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class CMGUnpooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_coarse, P):
        """
        Args:
            x_coarse: [Nc, F] tensor of coarse features
            P: [N, Nc] sparse or dense assignment matrix (one-hot rows)
        Returns:
            x_fine: [N, F] fine-level reconstructed features
        """
        if isinstance(P, list):
            P = torch.cat(P, dim=0)

        # Handle sparse matrix
        if P.is_sparse:
            return torch.sparse.mm(P, x_coarse)
        else:
            return P @ x_coarse
