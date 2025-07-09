
import os
import json
import pandas as pd
from pathlib import Path

LOG_DIR = Path("logs/autoencode")

def parse_run(folder):
    try:
        parts = folder.name.split("_")
        dataset = parts[0]
        k = int(parts[4][1:])  # e.g. 'k10' -> 10
        unpool = parts[5]
        pool = parts[6]
    except Exception:
        return None

    metrics_path = folder / "metrics.json"
    if not metrics_path.exists():
        return None

    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
        mse_final = metrics["losses"][-1]
        cos_final = metrics["cos_sims"][-1]
    except Exception:
        return None

    return {
        "dataset": dataset,
        "k": k,
        "pooling": pool,
        "unpooling": unpool,
        "final_mse": mse_final,
        "final_cos_sim": cos_final,
        "folder": str(folder)
    }

def main():
    all_runs = []
    for subdir in LOG_DIR.iterdir():
        if subdir.is_dir():
            result = parse_run(subdir)
            if result:
                all_runs.append(result)

    df = pd.DataFrame(all_runs)
    df = df.sort_values(by=["dataset", "k", "pooling", "unpooling"])
    df.to_csv("logs/autoencode_summary.csv", index=False)
    print(f"Saved summary to logs/autoencode_summary.csv")
    print(df.head(10))

if __name__ == "__main__":
    main()
