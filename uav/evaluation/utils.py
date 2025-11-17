import os
import csv

import numpy as np
import pandas as pd

from uav.evaluation.models import Metric, MeasurementDataBlock, EvaluationResult


def parse_results(metrics_csv_filepath: str, n_splits: int = 5) -> np.ndarray:
    """Parses the metrics csv file and averages every n_fold CV."""
    df = pd.read_csv(metrics_csv_filepath)
    
    metrics = df[['precision', 'recall', 'mAP50', 'mAP50-95']]
    avg_metrics = metrics.groupby(metrics.index // n_splits).mean()

    return np.asarray(avg_metrics)

def append_eval_results(results_csv_filepath: str, evaluation_result: EvaluationResult):
    """Appends a single evaluation result to a CSV file in a clean, flat format."""
    
    file_exists = os.path.exists(results_csv_filepath)
    data_row = [
        evaluation_result.implementation,
        evaluation_result.measured_metric.name,
        evaluation_result.friedman_p,
        *evaluation_result.wilcoxon_ps,
        *evaluation_result.hommel_ps,
        evaluation_result.effect_size
    ]
    
    with open(results_csv_filepath, "a", newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            header = [
                "implementation",
                "measured_metric",
                "friedman_p",
                "wilcoxon_p_1v2",
                "wilcoxon_p_2v3",
                "wilcoxon_p_1v3",
                "hommel_p_1v2",
                "hommel_p_2v3",
                "hommel_p_1v3",
                "effect_size"
            ]
            writer.writerow(header)
            
        writer.writerow(data_row)


def data_generator(metrics_csv_filepath: str, n_splits: int = 5):
    """Provides metric-wise measurement vectors."""
    results = parse_results(metrics_csv_filepath, n_splits)
    experiment_vz = results[:25]
    experiment_ir = results[25:50]
    experiment_hy = results[50:]

    for metric_idx, metric_name in enumerate(Metric): 
        yield MeasurementDataBlock(metric_name, experiment_vz[:, metric_idx], experiment_ir[:, metric_idx] , experiment_hy[:, metric_idx])
