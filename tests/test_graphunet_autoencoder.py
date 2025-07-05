import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GraphUNet

def test_graphunet_autoencoder_basic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root="experiments/data/Cora", name="Cora")
    data = dataset[0].to(device)

    class GraphUNetAutoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = GCNConv(dataset.num_node_features, 64)
            self.unet = GraphUNet(64, 64, 64, depth=1, pool_ratios=0.5)
            self.decoder = torch.nn.Linear(64, dataset.num_node_features)

        def forward(self, x, edge_index):
            x = F.relu(self.encoder(x, edge_index))
            x = self.unet(x, edge_index)
            x = self.decoder(x)
            return x

    model = GraphUNetAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(5):
        optimizer.zero_grad()
        x_hat = model(data.x, data.edge_index)
        loss = F.mse_loss(x_hat, data.x)
        loss.backward()
        optimizer.step()

    # Final check
    model.eval()
    x_hat = model(data.x, data.edge_index).detach()
    assert x_hat.shape == data.x.shape, f"Output shape {x_hat.shape} doesn't match input {data.x.shape}"
    assert not torch.isnan(x_hat).any(), "Model output contains NaNs"
    print("✅ GraphUNet autoencoder test passed.")
