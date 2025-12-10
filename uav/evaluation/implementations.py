import subprocess
import csv
import io

import pandas as pd
import pingouin as pg
from scipy.stats import friedmanchisquare as scipy_friedman, wilcoxon as scipy_wilcoxon
from statsmodels.stats.multitest import multipletests as statsmodels_multipletests

from uav.evaluation.models import ImplementationEvaluation, EvaluationResult, Metric
from uav.evaluation.scripts.STAC_statistical_analysis import fixed_friedman_test as stac_friedman


R_SCRIPT_FILEPATH = "uav/evaluation/scripts/statistical_analysis.r"


class SciPyEvaluation(ImplementationEvaluation):
    def evaluate(self) -> EvaluationResult:
        measured_metric = self.measurement_data_block.measured_metric
        measurements_one = self.measurement_data_block.measurements_one
        measurements_two = self.measurement_data_block.measurements_two
        measurements_three = self.measurement_data_block.measurements_three

        friedman_p: float = -1.0
        wilcoxon_ps: list[float] = [-1.0, -1.0, -1.0]
        hommel_ps: list[float] = [-1.0, -1.0, -1.0]

        _, friedman_p = scipy_friedman(measurements_one, measurements_two, measurements_three)
                    
        _, wilcoxon_ps[0] =  scipy_wilcoxon(x=measurements_one, y=measurements_two, method='exact')  
        _, wilcoxon_ps[1] =  scipy_wilcoxon(x=measurements_two, y=measurements_three, method='exact')
        _, wilcoxon_ps[2] =  scipy_wilcoxon(x=measurements_one, y=measurements_three, method='exact')

        return EvaluationResult("scipy", measured_metric, friedman_p, wilcoxon_ps, hommel_ps)

class REvaluation(ImplementationEvaluation):
    def evaluate(self) -> EvaluationResult:
        target_metric = self.measurement_data_block.measured_metric

        try:
            result = subprocess.run(
                ["Rscript", R_SCRIPT_FILEPATH],
                capture_output=True,
                text=True,
                check=True 
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error running R script:\n{e.stderr}")
        
        csv_output = result.stdout
        
        f = io.StringIO(csv_output.strip())
        reader = csv.DictReader(f)

        for row in reader:
            row_metric = Metric(row['measured_metric'])
            if row_metric == target_metric:
                wilcox_list = [
                    float(row['wilcoxon_p_1v2']),
                    float(row['wilcoxon_p_2v3']),
                    float(row['wilcoxon_p_1v3'])
                ]

                hommel_list = [
                    float(row['hommel_p_1v2']),
                    float(row['hommel_p_2v3']),
                    float(row['hommel_p_1v3'])
                ]

                return EvaluationResult(
                    implementation=row['implementation'],
                    measured_metric=row_metric,
                    friedman_p=float(row['friedman_p']),
                    wilcoxon_ps=wilcox_list,
                    hommel_ps=hommel_list,
                )

class StatsmodelsHommelwithScipyEvaluation(ImplementationEvaluation):
    def evaluate(self) -> EvaluationResult:
        measured_metric = self.measurement_data_block.measured_metric

        friedman_p: float = -1.0
        wilcoxon_ps: list[float] = [-1.0, -1.0, -1.0]
        hommel_ps: list[float] = [-1.0, -1.0, -1.0]

        start_result = SciPyEvaluation(self.measurement_data_block).evaluate()

        _, hommel_ps_array, _, _ = statsmodels_multipletests(start_result.wilcoxon_ps, alpha=self.alpha, method='hommel')
        hommel_ps = hommel_ps_array.tolist()

        return EvaluationResult("statsmodels_scipy", measured_metric, friedman_p, wilcoxon_ps, hommel_ps)

class StatsmodelsHommelwithREvaluation(ImplementationEvaluation):
    def evaluate(self) -> EvaluationResult:
        measured_metric = self.measurement_data_block.measured_metric

        friedman_p: float = -1.0
        wilcoxon_ps: list[float] = [-1.0, -1.0, -1.0]
        hommel_ps: list[float] = [-1.0, -1.0, -1.0]

        start_result = REvaluation(self.measurement_data_block).evaluate()

        _, hommel_ps_array, _, _ = statsmodels_multipletests(start_result.wilcoxon_ps, alpha=self.alpha, method='hommel')
        hommel_ps = hommel_ps_array.tolist()

        return EvaluationResult("statsmodels_r", measured_metric, friedman_p, wilcoxon_ps, hommel_ps)

class StatsmodelsHommelwithPinguoinEvaluation(ImplementationEvaluation):
    def evaluate(self) -> EvaluationResult:
        measured_metric = self.measurement_data_block.measured_metric

        friedman_p: float = -1.0
        wilcoxon_ps: list[float] = [-1.0, -1.0, -1.0]
        hommel_ps: list[float] = [-1.0, -1.0, -1.0]

        start_result = PinguoinEvaluation(self.measurement_data_block).evaluate()

        _, hommel_ps_array, _, _ = statsmodels_multipletests(start_result.wilcoxon_ps, alpha=self.alpha, method='hommel')
        hommel_ps = hommel_ps_array.tolist()

        return EvaluationResult("statsmodels_pinguoin", measured_metric, friedman_p, wilcoxon_ps, hommel_ps)


class PinguoinEvaluation(ImplementationEvaluation):
    def evaluate(self) -> EvaluationResult:
        measured_metric = self.measurement_data_block.measured_metric
        measurements_one = self.measurement_data_block.measurements_one
        measurements_two = self.measurement_data_block.measurements_two
        measurements_three = self.measurement_data_block.measurements_three

        friedman_p: float = -1.0
        wilcoxon_ps: list[float] = [-1.0, -1.0, -1.0]
        hommel_ps: list[float] = [-1.0, -1.0, -1.0]

        df = pd.DataFrame({
            'v1': {i: v for i, v in enumerate(measurements_one)},
            'v2': {i: v for i, v in enumerate(measurements_two)},
            'v3': {i: v for i, v in enumerate(measurements_three)},
        })

        friedman_p = pg.friedman(data=df, method='f')['p-unc'].values[0]
                    
        wilcoxon_ps[0] =  pg.wilcoxon(measurements_one, measurements_two)['p-val'].values[0]
        wilcoxon_ps[1] =  pg.wilcoxon(measurements_two, measurements_three)['p-val'].values[0]
        wilcoxon_ps[2] =  pg.wilcoxon(measurements_one, measurements_three)['p-val'].values[0]

        return EvaluationResult("pinguoin", measured_metric, friedman_p, wilcoxon_ps, hommel_ps)

class STACEvaluation(ImplementationEvaluation):
    def evaluate(self) -> EvaluationResult:
        measured_metric = self.measurement_data_block.measured_metric
        measurements_one = self.measurement_data_block.measurements_one
        measurements_two = self.measurement_data_block.measurements_two
        measurements_three = self.measurement_data_block.measurements_three

        friedman_p: float = -1.0
        wilcoxon_ps: list[float] = [-1.0, -1.0, -1.0]
        hommel_ps: list[float] = [-1.0, -1.0, -1.0]

        _, friedman_p, _, _ = stac_friedman(measurements_one, measurements_two, measurements_three)

        return EvaluationResult("stac", measured_metric, friedman_p, wilcoxon_ps, hommel_ps)

