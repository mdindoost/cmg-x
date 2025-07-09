import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_logs(log_dir="logs/structure_eval/", dataset_filter=None):
    logs = []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".json"):
            continue
        if dataset_filter and dataset_filter not in fname:
            continue
        with open(os.path.join(log_dir, fname), "r") as f:
            data = json.load(f)
            data["__file__"] = fname
            logs.append(data)
    return logs

def annotate(ax, config):
    ds = config["dataset"]
    k = config["k"]
    d = config["d"]
    t = config["threshold"]
    ax.set_title(f"{ds} | k={k}, d={d}, t={t}")

def plot_phi_hist(log, save_dir):
    config = log["config"]
    phi_dict = log["conductance"]["phi"]
    values = [v for v in phi_dict.values() if isinstance(v, (float, int)) and np.isfinite(v)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=30, color='steelblue', edgecolor='black')
    annotate(ax, config)
    ax.set_xlabel("φ")
    ax.set_ylabel("Number of Clusters")
    ax.grid(True)
    fig.tight_layout()
    outpath = os.path.join(save_dir, f"{config['dataset']}_k{config['k']}_phi_hist.png")
    fig.savefig(outpath)
    plt.close(fig)

def plot_spectrum(log, save_dir):
    config = log["config"]
    lamb_full = log["spectrum"].get("lambda_full", [])
    lamb_restricted = log["spectrum"].get("lambda_restricted", [])
    fig, ax = plt.subplots(figsize=(6, 4))
    if lamb_full:
        ax.plot(lamb_full, 'o-', label="λ (Full Spectrum)")
    if lamb_restricted:
        ax.plot(lamb_restricted, 'x-', label="λ (Restricted)")
    annotate(ax, config)
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    outpath = os.path.join(save_dir, f"{config['dataset']}_k{config['k']}_spectrum.png")
    fig.savefig(outpath)
    plt.close(fig)

def plot_cluster_sizes(log, save_dir):
    config = log["config"]
    sizes = log.get("cluster_sizes", [])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sizes, bins=30, color='purple', edgecolor='black')
    annotate(ax, config)
    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    fig.tight_layout()
    outpath = os.path.join(save_dir, f"{config['dataset']}_k{config['k']}_clustersize.png")
    fig.savefig(outpath)
    plt.close(fig)

def main():
    save_dir = "logs/structure_eval/plots"
    os.makedirs(save_dir, exist_ok=True)
    logs = load_logs()

    for log in logs:
        if "config" not in log:
            print(f"[SKIP] {log.get('__file__')} has no config.")
            continue
        plot_phi_hist(log, save_dir)
        plot_spectrum(log, save_dir)
        plot_cluster_sizes(log, save_dir)


if __name__ == "__main__":
    main()
