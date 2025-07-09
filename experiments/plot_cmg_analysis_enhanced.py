import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_all_logs(log_dir="logs/structure_eval/"):
    files = sorted([f for f in os.listdir(log_dir) if f.endswith(".json")])
    all_data = []
    for fname in files:
        with open(os.path.join(log_dir, fname), "r") as f:
            data = json.load(f)
            data["__file__"] = fname
            all_data.append(data)
    print(f"[LOADED] {len(all_data)} log files")
    return all_data

def plot_phi_hist(phi_dict):
    values = [v for k, v in phi_dict.items() if isinstance(v, (float, int)) and np.isfinite(v)]
    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=30, color='steelblue', edgecolor='black')
    plt.title("Histogram of Cluster Conductance (φ)")
    plt.xlabel("φ")
    plt.ylabel("Number of Clusters")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_spectrum(spectrum_info):
    lamb_full = spectrum_info.get("lambda_full")
    lamb_restricted = spectrum_info.get("lambda_restricted")

    plt.figure(figsize=(8, 4))
    if lamb_full:
        plt.plot(lamb_full, 'o-', label="λ (Full Spectrum)")
    if lamb_restricted:
        plt.plot(lamb_restricted, 'x-', label="λ (Restricted Subspace)")
    plt.title("Spectrum Comparison")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cosine_vs_degree(degree_list, cosine_list):
    if len(degree_list) != len(cosine_list):
        print("Skipping cosine vs degree plot due to shape mismatch.")
        return
    plt.figure(figsize=(8, 4))
    plt.scatter(degree_list, cosine_list, alpha=0.6, edgecolors='k', linewidths=0.2)
    plt.title("Cosine Similarity vs. Node Degree")
    plt.xlabel("Node Degree")
    plt.ylabel("Cosine Similarity (X₀ vs. Reconstruction)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def summarize_across_runs(logs):
    avg_phis = []
    mses = []
    cosines = []
    num_clusters = []
    for run in logs:
        if "conductance" in run and "avg_phi" in run["conductance"]:
            avg_phis.append(run["conductance"]["avg_phi"])
        if "feature" in run:
            mses.append(run["feature"].get("mse", np.nan))
            cosines.append(run["feature"].get("cosine", np.nan))
        if "num_clusters" in run:
            num_clusters.append(run["num_clusters"])

    if not avg_phis:
        print("No summary data found.")
        return

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.hist(avg_phis, bins=20, color='orange', edgecolor='black')
    plt.title("Avg φ Across Runs")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.hist(mses, bins=20, color='crimson', edgecolor='black')
    plt.title("Feature MSE Across Runs")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.hist(cosines, bins=20, color='seagreen', edgecolor='black')
    plt.title("Feature Cosine Similarity")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    logs = load_all_logs()

    latest = logs[-1]
    print(f"[INFO] Using latest run: {latest['__file__']}")

    if "conductance" in latest and "phi" in latest["conductance"]:
        plot_phi_hist(latest["conductance"]["phi"])
    if "spectrum" in latest:
        plot_spectrum(latest["spectrum"])

    if "feature" in latest and "per_node_cosine" in latest["feature"]:
        degs = latest["feature"].get("node_degree", [])
        coss = latest["feature"].get("per_node_cosine", [])
        plot_cosine_vs_degree(degs, coss)

    summarize_across_runs(logs)

if __name__ == "__main__":
    main()
