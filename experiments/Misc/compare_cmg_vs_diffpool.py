import os
import glob
import json
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def find_latest_run_folder(log_dir, prefix):
    matches = sorted(glob.glob(os.path.join(log_dir, f"{prefix}_*")))
    if not matches:
        raise FileNotFoundError(f"No folders found with prefix '{prefix}_*' in {log_dir}")
    return matches[-1]

def load_run_outputs(run_dir):
    metrics_path = os.path.join(run_dir, "metrics.json")
    reco_path = os.path.join(run_dir, "reconstruction.pt")

    if not os.path.isfile(metrics_path) or not os.path.isfile(reco_path):
        raise FileNotFoundError(f"Missing files in {run_dir}. Expected metrics.json and reconstruction.pt")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    reco = torch.load(reco_path)

    return {
        "cosine_sim": metrics.get("cosine_sim", []),
        "mse_loss": metrics.get("mse_loss", []),
        "x": reco["x"],
        "x_hat": reco["x_hat"]
    }

def plot_comparison(dataset="Cora"):
    cmg_dir = find_latest_run_folder("logs/autoencode", f"{dataset}_cmg")
    diff_dir = find_latest_run_folder("logs/diffpool", f"{dataset}_diffpool")

    cmg = load_run_outputs(cmg_dir)
    diff = load_run_outputs(diff_dir)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].plot(cmg["mse_loss"], label="CMG", linewidth=2)
    axs[0, 0].plot(diff["mse_loss"], label="DiffPool", linewidth=2)
    axs[0, 0].set_title("MSE Loss Curve")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("MSE")
    # axs[0, 0].set_yscale("log")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(cmg["cosine_sim"], label="CMG", linewidth=2)
    axs[0, 1].plot(diff["cosine_sim"], label="DiffPool", linewidth=2)
    axs[0, 1].set_title("Cosine Similarity Curve")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("CosSim")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    err_cmg = ((cmg["x_hat"] - cmg["x"]) ** 2).sum(dim=1).cpu().numpy()
    err_diff = ((diff["x_hat"] - diff["x"]) ** 2).sum(dim=1).cpu().numpy()

    axs[1, 0].hist(err_cmg, bins=50, alpha=0.6, label="CMG")
    axs[1, 0].hist(err_diff, bins=50, alpha=0.6, label="DiffPool")
    axs[1, 0].set_title("Per-node L2 Reconstruction Error")
    axs[1, 0].set_xlabel("||x̂ - x||²")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_combined = torch.cat([cmg["x"], cmg["x_hat"], diff["x_hat"]], dim=0).cpu().numpy()
    X_2d = tsne.fit_transform(X_combined)
    n = cmg["x"].shape[0]

    axs[1, 1].scatter(X_2d[:n, 0], X_2d[:n, 1], label="Original", alpha=0.5, s=10)
    axs[1, 1].scatter(X_2d[n:2*n, 0], X_2d[n:2*n, 1], label="CMG", alpha=0.5, s=10)
    axs[1, 1].scatter(X_2d[2*n:, 0], X_2d[2*n:, 1], label="DiffPool", alpha=0.5, s=10)
    axs[1, 1].set_title("t-SNE: Original vs Reconstructed")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/{dataset}_compare_diffpool_vs_cmg.png"
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    plot_comparison("Cora")
