from autoencode_graphunet import run_experiment

def run_all_graphunet():
    datasets = ["Cora", "Citeseer", "Pubmed"]
    for name in datasets:
        print(f"\n🚀 Running GraphUNet on {name}...")
        run_experiment(dataset_name=name, root_dir="data")

if __name__ == '__main__':
    run_all_graphunet()

