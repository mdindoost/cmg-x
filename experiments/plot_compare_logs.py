import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Directory where logs are saved
log_dir = Path("logs")

# Load logs
logs = {
    "CMG φγ": pd.read_csv(log_dir / "cmg_phi_gamma_log.csv"),
    "TopKPool": pd.read_csv(log_dir / "topkpool_log.csv"),
    "GCN": pd.read_csv(log_dir / "gcn_baseline_log.csv"),
}

# Plot test accuracy over epochs
plt.figure(figsize=(10, 6))
for name, df in logs.items():
    plt.plot(df["epoch"], df["test_acc"], label=name)
plt.title("Test Accuracy on Cora")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/test_accuracy_comparison.png")
plt.show()

# Print summary table
summary = {
    model: {
        "Final Test Acc": df["test_acc"].iloc[-1],
        "Max Test Acc": df["test_acc"].max(),
        "Final Val Acc": df["val_acc"].iloc[-1],
        "Avg φ": df["phi_avg"].mean() if "phi_avg" in df.columns else 0.0,
        "Avg γ": df["gamma_avg"].mean() if "gamma_avg" in df.columns else 0.0,
        "Avg φγ Loss": df["phi_gamma_loss"].mean() if "phi_gamma_loss" in df.columns else 0.0,
    }
    for model, df in logs.items()
}

summary_df = pd.DataFrame(summary).T.round(4)
print("\n=== Summary Table ===")
print(summary_df)

# Save summary
summary_df.to_csv("logs/accuracy_summary.csv")
