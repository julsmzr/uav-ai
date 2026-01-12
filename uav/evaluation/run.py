import numpy as np
from scipy.stats import rankdata

from uav.evaluation.utils import data_generator, append_eval_results, write_effect_size_results, write_metric_results, clean_csv_file
from uav.evaluation.models import EffectSizeResult, MetricResult, Metric
from uav.evaluation.implementations import SciPyEvaluation, REvaluation, PinguoinEvaluation, STACEvaluation, StatsmodelsHommelwithScipyEvaluation, StatsmodelsHommelwithPinguoinEvaluation


def calculate_eta_squared(vectors: list, rank_transform: bool = True) -> float:
    """Calculates Eta Squared (n2) for repeated measures."""
    data = np.column_stack(vectors)
    
    if rank_transform:
        data = rankdata(data, axis=1)
        
    n_subjects, _ = data.shape
    
    grand_mean = np.mean(data)
    group_means = np.mean(data, axis=0)  

    ss_effect = n_subjects * np.sum((group_means - grand_mean)**2)
    ss_total = np.sum((data - grand_mean)**2)
    
    if ss_total == 0:
        return 0.0
    return ss_effect / ss_total

def run_full_evaluation(
    metrics_csv_filepath: str, 
    stat_results_csv_filepath: str, 
    eff_results_csv_filepath: str, 
    metric_results_csv_filepath: str,
    n_splits: int = 5, 
    alpha: float = 0.05,
    force_recreate: bool = False
    ) -> None:
    """Run full evaluation pipeline for all implementations."""
    to_evaluate = [
        SciPyEvaluation,
        REvaluation,
        PinguoinEvaluation,
        STACEvaluation,
        StatsmodelsHommelwithScipyEvaluation,
        StatsmodelsHommelwithPinguoinEvaluation
    ]

    effect_sizes: list[EffectSizeResult] = []
    metric_results: list[MetricResult] = []

    if force_recreate:
        clean_csv_file(stat_results_csv_filepath)
        clean_csv_file(eff_results_csv_filepath)
        clean_csv_file(metric_results_csv_filepath)

    for measurement_data_block in data_generator(metrics_csv_filepath, n_splits):
        for Implementation in to_evaluate:
            append_eval_results(stat_results_csv_filepath, Implementation(measurement_data_block, alpha).evaluate())
    
        experiments = {
            "VZ": measurement_data_block.measurements_one,
            "IR": measurement_data_block.measurements_two,
            "HY": measurement_data_block.measurements_three
        }
        
        ne2 = calculate_eta_squared(
                    vectors=[experiments["VZ"], experiments["IR"], experiments["HY"]], 
                    rank_transform=True
                )

        effect_sizes.append(
            EffectSizeResult(
                measurement_data_block.measured_metric, 
                ne2
            )
        )

        for exp_label, data in experiments.items():
            metric_results.append(
                 MetricResult(
                    experiment=exp_label, 
                    metric=measurement_data_block.measured_metric, 
                    mean=np.mean(data), 
                    std=np.std(data)
                )
            )

    write_effect_size_results(eff_results_csv_filepath, effect_sizes)
    write_metric_results(metric_results_csv_filepath, metric_results)
