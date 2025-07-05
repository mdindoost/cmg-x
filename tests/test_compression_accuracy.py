import torch
import torch.nn.functional as F
from cmgx.pyg_pool import CMGPooling
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv


def test_cmg_pooling_compression_and_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root="/home/mohammad/cmg-x/data/Cora", name="Cora")
    data = dataset[0].to(device)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    class CMGClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(dataset.num_node_features, 64)
            self.pool = CMGPooling()
            self.conv2 = GCNConv(64, dataset.num_classes)

        def forward(self, x, edge_index, batch):
            x = F.relu(self.conv1(x, edge_index))
            results = self.pool(x, edge_index, batch, return_P=True)
            x_c, edge_index_c, batch_c, P, _ = results
            x_out = self.conv2(x_c, edge_index_c)
            return x_out, P

    model = CMGClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for _ in range(100):
        model.train()
        optimizer.zero_grad()
        out, P = model(data.x, data.edge_index, data.batch)

        # Prepare coarse labels using only training labels
        y = data.y
        y_onehot = F.one_hot(y, num_classes=dataset.num_classes).float()

        if isinstance(P, list):
            P = torch.cat(P, dim=0)

        y_coarse = (P.T @ y_onehot).argmax(dim=1)
        loss = F.cross_entropy(out, y_coarse)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        out, P = model(data.x, data.edge_index, data.batch)
        if isinstance(P, list):
            P = torch.cat(P, dim=0)

        out_fine = P @ out  # project coarse logits back
        pred = out_fine.argmax(dim=1)
        acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        compression_ratio = P.shape[1] / P.shape[0]

    print(f"Test Acc: {acc:.4f} | Compression: {compression_ratio:.3f}")
    assert acc > 0.6, f"CMG accuracy too low: {acc:.4f}"
    assert compression_ratio < 1.0, f"CMG did not compress: {compression_ratio:.3f}"
