from autoencode_cmg_unpool_variants import run_experiment

def run_all_cmg_unpool_experiments():
    datasets = ["Cora", "Citeseer", "Pubmed"]
    pooling_modes = ["mean", "sum"]
    unpooling_methods = ["copy", "central", "first", "random"]

    for dataset in datasets:
        for pool_mode in pooling_modes:
            for unpool_mode in unpooling_methods:
                run_name = f"{dataset}_cmg_{pool_mode}_unpool_{unpool_mode}_original"
                print(f"\n🚀 Running CMG autoencoder: {run_name}")
                run_experiment(
                    dataset_name=dataset,
                    out_dir="logs/autoencode",
                    epochs=200,
                    pooling_mode=pool_mode,
                    unpooling_mode=unpool_mode
                )

if __name__ == '__main__':
    run_all_cmg_unpool_experiments()
