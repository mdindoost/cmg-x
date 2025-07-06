
import os
import itertools

datasets = ["PROTEINS", "NCI1", "ENZYMES", "IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY"]

models = ["gcn", "topk", "diffpool", "sagpool", "asap", "cmg"]
seeds = [42, 1337, 2024, 777, 0]

for dataset, model, seed in itertools.product(datasets, models, seeds):
    print(f"🚀 Running: dataset={dataset}, model={model}, seed={seed}")

    if model == "cmg":
        cmd = f"python train_cmg.py --dataset {dataset} --seed {seed}"
    else:
        cmd = f"python train_baseline_updated.py --dataset {dataset} --model {model} --seed {seed}"

    os.system(cmd)
