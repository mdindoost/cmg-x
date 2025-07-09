
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_summary(csv_path="logs/autoencode_summary.csv", output_dir="logs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Plot MSE for each dataset vs. k
    for dataset in df['dataset'].unique():
        df_d = df[df['dataset'] == dataset]
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_d, x='k', y='final_mse', hue='unpooling')
        plt.title(f"{dataset} - Final MSE vs k (by Unpooling)")
        plt.xlabel("Filter k")
        plt.ylabel("Final MSE")
        plt.legend(title="Unpooling")
        plt.grid(True)
        plt.savefig(f"{output_dir}/{dataset}_mse_vs_k_unpooling.png")

    # Plot Cosine Similarity
    for dataset in df['dataset'].unique():
        df_d = df[df['dataset'] == dataset]
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_d, x='k', y='final_cos_sim', hue='unpooling')
        plt.title(f"{dataset} - Cosine Similarity vs k (by Unpooling)")
        plt.xlabel("Filter k")
        plt.ylabel("Final Cosine Similarity")
        plt.legend(title="Unpooling")
        plt.grid(True)
        plt.savefig(f"{output_dir}/{dataset}_cos_sim_vs_k_unpooling.png")

    # Optional: bar plot for all config scores
    for dataset in df['dataset'].unique():
        df_d = df[df['dataset'] == dataset].copy()
        df_d['config'] = df_d['k'].astype(str) + "_" + df_d['pooling'] + "_" + df_d['unpooling']
        df_d = df_d.sort_values(by="final_mse")

        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_d, x='config', y='final_mse')
        plt.xticks(rotation=90)
        plt.title(f"{dataset} - MSE by Configuration")
        plt.ylabel("Final MSE")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{dataset}_mse_barplot.png")

if __name__ == "__main__":
    plot_summary()
