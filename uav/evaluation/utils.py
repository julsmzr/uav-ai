import os
import csv
from contextlib import contextmanager

import numpy as np
import pandas as pd

from uav.evaluation.models import Metric, MeasurementDataBlock, EvaluationResult, EffectSizeResult, MetricResult


def parse_results(metrics_csv_filepath: str, n_splits: int = 5) -> np.ndarray:
    """Parses the metrics csv file and averages every n_fold CV."""
    df = pd.read_csv(metrics_csv_filepath)
    
    metrics = df[['precision', 'recall', 'mAP50', 'mAP50-95']]
    avg_metrics = metrics.groupby(metrics.index // n_splits).mean()

    return np.asarray(avg_metrics)

@contextmanager
def csv_appender(filepath: str, header: list[str]):
    """Clean csv appending, writes header if empty or file does not exist."""
    file_exists = os.path.exists(filepath)
    is_empty = file_exists and os.path.getsize(filepath) == 0

    with open(filepath, "a", newline='') as f:
        writer = csv.writer(f)

        if not file_exists or is_empty:
            writer.writerow(header)
            
        yield writer

def append_eval_results(results_csv_filepath: str, evaluation_result: EvaluationResult):
    """Appends a single evaluation result to a CSV file in a clean, flat format."""
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
    ]

    with csv_appender(results_csv_filepath, header) as writer:
        writer.writerow([
            evaluation_result.implementation,
            evaluation_result.measured_metric.name,
            evaluation_result.friedman_p,
            *evaluation_result.wilcoxon_ps,
            *evaluation_result.hommel_ps,
        ])

def write_effect_size_results(results_csv_filepath: str, effect_sizes: list[EffectSizeResult]):
    """Writes effect size results to csv file."""
    header = [
        "measured_metric",
        "effect_size",
    ]

    with csv_appender(results_csv_filepath, header) as writer:

        for effect_size in effect_sizes:
            writer.writerow([
                effect_size.metric.name,
                effect_size.effect_size
            ])

def write_metric_results(results_csv_filepath: str, metrics: list[MetricResult]):
    """Writes effect size results to csv file."""
    header = [
        "experiment",
        "measured_metric",
        "mean",
        "std",
    ]

    with csv_appender(results_csv_filepath, header) as writer:

        for metric in metrics:
            writer.writerow([
                metric.experiment,
                metric.metric.name,
                metric.mean,
                metric.std
            ])

def clean_csv_file(csv_filepath: str) -> None:
    if os.path.exists(csv_filepath):
        os.remove(csv_filepath)

def data_generator(metrics_csv_filepath: str, n_splits: int = 5):
    """Provides metric-wise measurement vectors."""
    results = parse_results(metrics_csv_filepath, n_splits)
    experiment_vz = results[:25]
    experiment_ir = results[25:50]
    experiment_hy = results[50:]

    for metric_idx, metric_name in enumerate(Metric): 
        yield MeasurementDataBlock(metric_name, experiment_vz[:, metric_idx], experiment_ir[:, metric_idx] , experiment_hy[:, metric_idx])
