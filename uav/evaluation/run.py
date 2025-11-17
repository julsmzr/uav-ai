

from uav.evaluation.utils import data_generator, append_eval_results
from uav.evaluation.implementations import SciPyEvaluation

def run_full_evaluation(metrics_csv_filepath: str, results_csv_filepath: str, n_splits: int = 5) -> None:

    for measurement_data_block in data_generator(metrics_csv_filepath, n_splits):
        
        scipy_res = SciPyEvaluation(measurement_data_block).evaluate()
        append_eval_results(results_csv_filepath, scipy_res)



# STAC
# https://github.com/citiususc/stac?tab=readme-ov-file
# https://tec.citius.usc.es/stac/doc/

# TODO: do this once with scipy, once with statsmodels, once with SCAT library, 
# then run as subprocess the R script as well. each should write results to a file. 
# then use matplotlib to plot differences in s.a. results.

# also rework visualization pipeline to render multiple plots (start in setup script end)
# violin plots for P,R,.. for measurements, then for different s.a. libs find vis strategy
