
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

log_dir = "/home/mohammad/cmg-x/experiments/logs/gnn"
datasets = ["PROTEINS", "NCI1", "ENZYMES"]
models = ["gcn", "topk", "diffpool", "cmg"]

# Aggregated storage: {dataset -> {model -> list of dicts (each a run)}}
results = defaultdict(lambda: defaultdict(list))

for fname in os.listdir(log_dir):
    if not fname.endswith(".json"):
        continue
    parts = fname.replace(".json", "").split("_")
    if len(parts) < 3:
        continue
    dataset, model, seed = parts[0], parts[1], parts[2]
    if dataset not in datasets or model not in models:
        continue

    with open(os.path.join(log_dir, fname)) as f:
        log = json.load(f)
        results[dataset][model].append(log)

# Plotting helper
def plot_metric(metric_key, ylabel, filename, normalize_epochs=True):
    for dataset in datasets:
        plt.figure()
        for model in models:
            logs = results[dataset][model]
            # Only include runs that contain the metric
            valid_logs = [run for run in logs if metric_key in run]
            if not valid_logs:
                print(f"⚠️ Skipping {dataset} {model} — no {metric_key}")
                continue

            max_len = max(len(run[metric_key]) for run in valid_logs)
            data = np.zeros((len(logs), max_len))
            for i, run in enumerate(logs):
                length = len(run[metric_key])
                padded = run[metric_key] + [run[metric_key][-1]] * (max_len - len(run[metric_key]))
                data[i, :] = padded[:max_len]
            mean_vals = np.mean(data, axis=0)
            std_vals = np.std(data, axis=0)

            epochs = np.arange(max_len)
            if normalize_epochs:
                plt.plot(epochs, mean_vals, label=model.upper())
                plt.fill_between(epochs, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)
            else:
                plt.bar(model.upper(), mean_vals[-1], yerr=std_vals[-1], capsize=5)

        plt.title(f"{metric_key.replace('_', ' ').capitalize()} - {dataset}")
        plt.xlabel("Epoch" if normalize_epochs else "Model")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        outname = f"{log_dir}/{dataset}_{metric_key}_{'epochs' if normalize_epochs else 'final'}.png"
        plt.savefig(outname)
        print(f"Saved: {outname}")

# Generate plots
plot_metric("train_acc", "Training Accuracy", "train_acc.png")
plot_metric("val_acc", "Validation Accuracy", "val_acc.png")
plot_metric("test_acc", "Test Accuracy", "test_acc.png")
plot_metric("cos_sim", "Cosine Similarity", "cos_sim.png")
plot_metric("mse", "MSE", "mse.png")
plot_metric("compression", "Compression Ratio", "compression.png")
