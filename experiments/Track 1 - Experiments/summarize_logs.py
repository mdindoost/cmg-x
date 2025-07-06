
import os
import json
import numpy as np
from collections import defaultdict

log_dir = "/home/mohammad/cmg-x/experiments/logs/gnn"
datasets = ["PROTEINS", "NCI1", "ENZYMES"]
models = ["gcn", "topk", "diffpool", "cmg"]

def summarize_runs(metric_list):
    vals = np.array(metric_list)
    mean = np.mean(vals)
    std = np.std(vals)
    return f"{mean*100:.2f} ± {std*100:.2f}"

# Collect all logs
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
        final_test = log['test_acc'][-1]
        entry = {'test_acc': final_test}
        if model == "cmg":
            entry['cos_sim'] = log['cos_sim'][-1]
            entry['mse'] = log['mse'][-1]
            entry['compression'] = log['compression'][-1]
        results[dataset][model].append(entry)

# Print table
print("\n\n### Summary Table (Final Epoch Results)\n")
header = ["Dataset", "Model", "Test Acc", "CosSim", "MSE", "Compression"]
print("{:<10} {:<8} {:<15} {:<10} {:<10} {:<12}".format(*header))
print("-" * 70)

for dataset in datasets:
    for model in models:
        entries = results[dataset][model]
        if not entries:
            continue
        test_accs = [e['test_acc'] for e in entries]
        cos_sims = [e.get('cos_sim') for e in entries if 'cos_sim' in e]
        mses = [e.get('mse') for e in entries if 'mse' in e]
        compressions = [e.get('compression') for e in entries if 'compression' in e]

        row = [
            dataset,
            model.upper(),
            summarize_runs(test_accs),
            summarize_runs(cos_sims) if cos_sims else "–",
            summarize_runs(mses) if mses else "–",
            summarize_runs(compressions) if compressions else "–",
        ]
        print("{:<10} {:<8} {:<15} {:<10} {:<10} {:<12}".format(*row))
