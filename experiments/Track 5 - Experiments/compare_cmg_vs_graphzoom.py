import os
import json
import pandas as pd
from glob import glob

def load_metrics(log_root):
    data = []
    for metrics_path in glob(os.path.join(log_root, "*", "metrics.json")):
        graph_name = os.path.basename(os.path.dirname(metrics_path))
        try:
            with open(metrics_path) as f:
                m = json.load(f)
                data.append({
                    "graph": graph_name,
                    "nodes": m["num_nodes"],
                    "edges": m["num_edges"],
                    "clusters": m["num_clusters"],
                    "conductance": round(m["conductance"], 4),
                    "path_distortion": m.get("path_length_distortion", None),
                    "runtime_sec": m.get("runtime_seconds", None)
                })
        except Exception as e:
            print(f"⚠️ Skipped {graph_name}: {e}")
    return pd.DataFrame(data)

# Load logs
cmg_df = load_metrics("experiments/logs/structure_eval")
zoom_df = load_metrics("experiments/logs/structure_eval_graphzoom")

if cmg_df.empty and zoom_df.empty:
    print("❌ No metrics found for either CMG or GraphZoom.")
    exit()

# Suffix and merge
if not cmg_df.empty:
    cmg_df = cmg_df.add_suffix("_cmg").rename(columns={"graph_cmg": "graph"})
if not zoom_df.empty:
    zoom_df = zoom_df.add_suffix("_zoom").rename(columns={"graph_zoom": "graph"})

# Merge on graph name
if not cmg_df.empty and not zoom_df.empty:
    df = pd.merge(cmg_df, zoom_df, on="graph", how="outer")
else:
    df = cmg_df if not cmg_df.empty else zoom_df

# Keep only one copy of nodes/edges (from CMG side)
if "nodes_cmg" in df.columns:
    df = df.rename(columns={
        "nodes_cmg": "nodes",
        "edges_cmg": "edges"
    })

# Final column order
columns = [
    "graph", "nodes", "edges",
    "clusters_cmg", "conductance_cmg", "path_distortion_cmg", "runtime_sec_cmg",
    "clusters_zoom", "conductance_zoom", "path_distortion_zoom", "runtime_sec_zoom"
]
df = df[[col for col in columns if col in df.columns]]

# Sort and save
df = df.sort_values("graph")
csv_path = "experiments/logs/structure_eval_comparison.csv"
md_path = "experiments/logs/structure_eval_comparison.md"
df.to_csv(csv_path, index=False)
df.to_markdown(md_path, index=False)

# Print to terminal
print("\n📊 CMG vs GraphZoom Comparison:")
# Print selected summary to terminal
cols_to_display = [
    "graph",
    "nodes",
    "edges",
    "clusters_cmg",
    "path_distortion_cmg",
    "clusters_zoom",
    "path_distortion_zoom"
]
print("\n📊 CMG vs GraphZoom Comparison (Selected Columns):")
print(df[cols_to_display].to_string(index=False))
print(f"\n✅ Saved comparison table to:\n- {csv_path}\n- {md_path}")
