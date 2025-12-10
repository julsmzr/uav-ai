import numpy as np

from uav.evaluation.utils import data_generator

def analyze_metrics(filepath: str):
    """Calculates macro mean and std for each experiment group metrics (VZ, IR, HY)."""
    print(f"{'Metric':<15} | {'Experiment':<10} | {'Mean':<10} | {'Std Dev':<10}")
    print("-" * 50)

    for block in data_generator(filepath):
        metric_name = block.measured_metric
        vz_data = block.measurements_one
        ir_data = block.measurements_two
        hy_data = block.measurements_three

        experiments = {
            "VZ": vz_data,
            "IR": ir_data,
            "HY": hy_data
        }

        for exp_label, data in experiments.items():
            if data is not None and len(data) > 0:
                mean_val = np.mean(data)
                std_val = np.std(data)
                
                m_name = metric_name.value if hasattr(metric_name, 'value') else str(metric_name)
                
                print(f"{m_name:<15} | {exp_label:<10} | {mean_val:.4f} ± {std_val:.4f}")
        
        print("-" * 50)

