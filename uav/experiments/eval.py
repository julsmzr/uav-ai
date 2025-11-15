
import numpy as np
from enum import Enum

from scipy.stats import friedmanchisquare, wilcoxon


class Metrics(Enum):
    PRECISION = 1
    RECALL = 2
    mAP50 = 3
    mAP50_95 = 4 


def parse_results(metrics_txt_filepath: str, n_splits: int = 5) -> np.ndarray:
    data = []
    with open(metrics_txt_filepath, 'r') as f:
        for l in f.readlines():
            _, p, r, mAP50, mAP50_95 = l.split(",") 
            data.append([p, r, mAP50, mAP50_95])

    if not len(data) % n_splits == 0:
        print("Warning: metrics.txt does not contain all needed fold results.")
    
    # METRICS x VALUES
    data_array = np.asarray(data, dtype=float)
    
    # Average every n_splits consecutive results
    averaged_data = []
    for i in range(0, len(data_array), n_splits):
        if i + n_splits <= len(data_array):
            chunk = data_array[i:i + n_splits]
            averaged_chunk = np.mean(chunk, axis=0)
            averaged_data.append(averaged_chunk)
    
    return np.asarray(averaged_data)

def run_full_evaluation(metrics_txt_filepath: str, n_splits: int = 5) -> None:

    results = parse_results(metrics_txt_filepath, n_splits)
    experiment_vz = results[:25]
    experiment_ir = results[25:50]
    experiment_hy = results[50:]

    for metric_idx, metric_name in enumerate(Metrics):
        metric_vz = experiment_vz[:, metric_idx]
        metric_ir = experiment_ir[:, metric_idx] 
        metric_hy = experiment_hy[:, metric_idx]
        
        friedman_stat, friedman_p = friedmanchisquare(metric_vz, metric_ir, metric_hy)
        
        print(f"\n{metric_name.name}:")
        print(f"  Friedman test: χ² = {friedman_stat:.4f}, p = {friedman_p:.4f}")
        
        if friedman_p < 0.05:
            print("  * Significant difference detected - consider post-hoc tests")

if __name__ == "__main__":
    run_full_evaluation("results/metrics.txt")
