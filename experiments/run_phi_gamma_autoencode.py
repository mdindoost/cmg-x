import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from phi_gamma_autoencoder import PhiGammaAutoencoder

# Synthetic graph (replace with real later)
def generate_synthetic_graph(n=100, feat_dim=16):
    x = torch.randn(n, feat_dim)
    A = torch.randint(0, 2, (n, n)).float()
    A = torch.triu(A, diagonal=1)
    A = A + A.T
    A.fill_diagonal_(0)
    return x, A.to_sparse()

# Training loop
def train(model, x, adj, epochs=50, lambda_phi_gamma=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    recon_losses, phi_gamma_losses = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        x_hat, phi_gamma_loss, P, phi_tensor, gamma_tensor = model(x, adj)

        recon_loss = F.mse_loss(x_hat, x)
        total_loss = recon_loss + lambda_phi_gamma * phi_gamma_loss
        total_loss.backward()
        optimizer.step()

        recon_losses.append(recon_loss.item())
        phi_gamma_losses.append(phi_gamma_loss.item())

        if epoch % 10 == 0:
            print(f"[Epoch {epoch:03d}] Recon: {recon_loss.item():.4f}  PhiGamma: {phi_gamma_loss.item():.4f}")
            print(f"  φ avg: {phi_tensor.mean().item():.4f}  γ avg: {gamma_tensor.mean().item():.4f}")

    return recon_losses, phi_gamma_losses

if __name__ == "__main__":
    x, adj = generate_synthetic_graph(n=100, feat_dim=16)
    model = PhiGammaAutoencoder(
        in_channels=16,
        hidden_channels=32,
        num_clusters=10,
        recon_method='soft'  # can try 'copy', 'first', 'central'
    )

    recon_losses, phi_gamma_losses = train(model, x, adj, epochs=100)

    # Plot
    plt.plot(recon_losses, label="Reconstruction Loss")
    plt.plot(phi_gamma_losses, label="φγ Loss")
    plt.legend()
    plt.title("PhiGamma Autoencoder Losses")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("phi_gamma_training_curve.png")
    plt.show()
