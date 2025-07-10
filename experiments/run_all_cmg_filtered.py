import os
import itertools
from multiprocessing import Pool

datasets = ["Cora", "Citeseer", "Pubmed"]
# datasets = ["Citeseer"]
ks = [5, 10, 15]
unpool_modes = ["copy", "mean", "central", "first", "random"]
pooling_modes = ["sum", "mean"]

def run_job(args):
    dataset, k, unpool, pool = args
    cmd = f"python autoencode_cmg_filtered.py --dataset {dataset} --k {k} --unpooling_mode {unpool} --pooling_mode {pool}"
    print(f"[RUNNING] {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    jobs = list(itertools.product(datasets, ks, unpool_modes, pooling_modes))
    with Pool(processes=3) as pool:
        pool.map(run_job, jobs)