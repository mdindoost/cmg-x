import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

LOG_AUTOENCODE = "logs/autoencode"
LOG_DIFFPOOL = "logs/diffpool"

def parse_run_info(folder_name):
    parts = folder_name.lower().split("_")
    dataset = parts[0]
    method = "graphunet" if "graphunet" in folder_name.lower() else \
             "diffpool" if "diffpool" in folder_name.lower() else "cmg"
    feature_type = "identity" if "identity" in parts else "original"
    has_unpool = True if method == "graphunet" else ("unpool" in folder_name.lower() and "nounpool" not in folder_name.lower())
    return dataset, method, feature_type, has_unpool

def load_metrics(folder_path):
    metrics_file = os.path.join(folder_path, "metrics.json")
    if not os.path.exists(metrics_file):
        return None, None
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    losses = metrics.get("losses") or metrics.get("mse_loss", [])
    cossims = metrics.get("cos_sims") or metrics.get("cosine_sim", [])
    return (losses[-1] if losses else None), (cossims[-1] if cossims else None)

def find_autoencode_runs():
    entries = os.listdir(LOG_AUTOENCODE)
    valid_runs = []

    for folder in entries:
        path = os.path.join(LOG_AUTOENCODE, folder)
        if not os.path.isdir(path):
            continue
        dataset, method, feature_type, has_unpool = parse_run_info(folder)

        if feature_type != "original" or not has_unpool:
            continue

        valid_runs.append((dataset, method, feature_type, folder, path))

    # Keep latest per (dataset, method)
    latest = {}
    for dataset, method, feature_type, folder, path in sorted(valid_runs):
        key = (dataset, method)
        if key not in latest or folder > latest[key][0]:
            latest[key] = (folder, feature_type, path)

    results = []
    for (dataset, method), (folder, feature_type, path) in latest.items():
        mse, cos = load_metrics(path)
        if mse is not None and cos is not None:
            results.append({
                "Dataset": dataset.capitalize(),
                "Method": method,
                "FeatureType": feature_type,
                "FinalMSE": mse,
                "FinalCosSim": cos,
                "Folder": folder
            })
    return results

def find_diffpool_runs():
    entries = os.listdir(LOG_DIFFPOOL)
    latest = {}

    for folder in sorted(entries):
        path = os.path.join(LOG_DIFFPOOL, folder)
        if not os.path.isdir(path) or "diffpool" not in folder.lower():
            continue
        dataset = folder.lower().split("_")[0]
        if dataset not in latest or folder > latest[dataset][0]:
            latest[dataset] = (folder, path)

    results = []
    for dataset, (folder, path) in latest.items():
        mse, cos = load_metrics(path)
        if mse is not None and cos is not None:
            results.append({
                "Dataset": dataset.capitalize(),
                "Method": "diffpool",
                "FeatureType": "original",
                "FinalMSE": mse,
                "FinalCosSim": cos,
                "Folder": folder
            })
    return results

def main():
    results = find_autoencode_runs() + find_diffpool_runs()

    # Print table
    print(f"| Dataset | Method     | Feature | Final MSE | Final Cosine | Folder")
    print(f"|---------|------------|---------|-----------|--------------|--------")
    for r in sorted(results, key=lambda x: (x["Dataset"], x["Method"])):
        print(f"| {r['Dataset']:<8} | {r['Method']:<10} | {r['FeatureType']:<8} | "
              f"{r['FinalMSE']:<9.6f} | {r['FinalCosSim']:<12.4f} | {r['Folder']}")

    # Save plots
    save_dir = os.path.join(LOG_AUTOENCODE, "summary")
    os.makedirs(save_dir, exist_ok=True)

    def plot_metric(key, ylabel, filename):
        import numpy as np

        methods = ["cmg", "diffpool", "graphunet"]
        datasets = sorted(set(r["Dataset"] for r in results))
        method_colors = {
            "cmg": "#1f77b4",        # blue
            "diffpool": "#ff7f0e",   # orange
            "graphunet": "#2ca02c"   # green
        }

        # Build grouped data matrix
        bar_values = []
        for method in methods:
            bar_values.append([
                next((r[key] for r in results if r["Dataset"] == dataset and r["Method"] == method), 0)
                for dataset in datasets
            ])

        x = np.arange(len(datasets))  # label positions
        width = 0.25

        plt.figure(figsize=(8, 5))
        for i, method in enumerate(methods):
            offset = (i - 1) * width
            plt.bar(x + offset, bar_values[i], width=width,
                    label=method.capitalize(), color=method_colors[method])

        plt.xticks(x, datasets)
        plt.ylabel(ylabel)
        plt.title(f"Autoencoder Comparison: {ylabel}")
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        out_path = os.path.join(LOG_AUTOENCODE, "summary", filename)
        plt.savefig(out_path)
        print(f"✅ Saved grouped plot: {filename}")


    plot_metric("FinalMSE", "Final MSE", "compare_mse.png")
    plot_metric("FinalCosSim", "Cosine Similarity", "compare_cosine.png")

if __name__ == "__main__":
    main()
