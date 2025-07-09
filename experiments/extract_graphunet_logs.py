import os
import json
import pandas as pd
from pathlib import Path

LOG_DIR = Path("logs/autoencode")

def extract_graphunet_logs(log_dir=LOG_DIR):
    all_runs = []

    for folder in log_dir.iterdir():
        if not folder.is_dir():
            continue

        folder_name = folder.name.lower()
        if "graphunet" not in folder_name:
            continue

        parts = folder.name.split("_")
        if len(parts) < 2:
            continue

        dataset = parts[0]
        pooling = "graphunet"
        unpooling = "n/a"
        k = None  # not used in GraphUNet

        metrics_path = folder / "metrics.json"
        if not metrics_path.exists():
            print(f"[SKIP] No metrics found in {folder}")
            continue

        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            mse_final = metrics["losses"][-1]
            cos_final = metrics["cos_sims"][-1]
        except Exception as e:
            print(f"[WARN] Failed to read metrics in {folder}: {e}")
            continue

        all_runs.append({
            "dataset": dataset,
            "k": k,
            "pooling": pooling,
            "unpooling": unpooling,
            "final_mse": mse_final,
            "final_cos_sim": cos_final,
            "folder": str(folder)
        })

    return pd.DataFrame(all_runs)


if __name__ == "__main__":
    df = extract_graphunet_logs()
    if df.empty:
        print("⚠️ No GraphUNet runs found.")
    else:
        output_path = LOG_DIR / "graphunet_summary.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Saved GraphUNet summary to {output_path}")
        print(df.head())
