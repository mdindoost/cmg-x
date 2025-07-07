import torch
import torch.nn as nn
from cmgx.torch_interface import cmg_unpool_features

class CMGUnpooling(nn.Module):
    def __init__(self, method='copy'):
        """
        Args:
            method (str): Unpooling strategy:
                - 'copy' (default): broadcast coarse feature to all nodes in cluster
                - 'mean': divide by cluster size
                - 'first': assign feature to first node in cluster
                - 'random': assign to random node in cluster
                - 'central': assign to highest-degree node in cluster
        """
        super().__init__()
        self.method = method

    def forward(self, x_coarse, P, cluster_assignments=None, degree=None):
        if isinstance(P, list):
            P = P[0]

        return cmg_unpool_features(
            x_coarse, P,
            method=self.method,
            cluster_assignments=cluster_assignments,
            degree=degree
        )
