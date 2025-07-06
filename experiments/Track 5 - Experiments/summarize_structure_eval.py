import os
import json
import pandas as pd
from glob import glob

log_root = "experiments/logs/structure_eval"
out_csv = "experiments/logs/structure_eval_summary.csv"
out_md = "experiments/logs/structure_eval_summary.md"

data = []

for metrics_path in glob(os.path.join(log_root, "*", "metrics.json")):
    name = os.path.basename(os.path.dirname(metrics_path))
    with open(metrics_path) as f:
        m = json.load(f)
        data.append({
            "graph": name,
            "nodes": m["num_nodes"],
            "edges": m["num_edges"],
            "clusters": m["num_clusters"],
            "conductance": round(m["conductance"], 4)
        })

df = pd.DataFrame(data).sort_values(by="graph")
print(df.to_string(index=False))

df.to_csv(out_csv, index=False)
df.to_markdown(out_md, index=False)

print(f"\n✅ Saved summary to:\n- {out_csv}\n- {out_md}")
