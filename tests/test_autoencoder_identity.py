import torch
import torch.nn.functional as F
from cmgx.pyg_pool import CMGPooling
from cmgx.pyg_unpool import CMGUnpooling
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

def test_cmg_autoencoder_identity():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root="/tmp/Cora", name="Cora")
    data = dataset[0].to(device)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    class CMGAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = GCNConv(dataset.num_node_features, 64)
            self.pool = CMGPooling()
            self.decoder = GCNConv(64, dataset.num_node_features)
            self.unpool = CMGUnpooling()

        def forward(self, x, edge_index, batch):
            x = F.relu(self.encoder(x, edge_index))
            x_c, edge_index_c, batch_c, P, _ = self.pool(x, edge_index, batch, return_P=True)

            x_dec = self.decoder(x_c, edge_index_c)
            x_rec = self.unpool(x_dec, P)
            return x_rec

    model = CMGAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(100):
        model.train()
        optimizer.zero_grad()
        x_hat = model(data.x, data.edge_index, data.batch)
        loss = F.mse_loss(x_hat, data.x)
        loss.backward()
        optimizer.step()

    final_mse = F.mse_loss(x_hat, data.x).item()
    assert final_mse < 0.02, f"High reconstruction error: {final_mse:.4f}"
