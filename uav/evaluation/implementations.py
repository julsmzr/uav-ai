from scipy.stats import friedmanchisquare, wilcoxon

from uav.evaluation.models import ImplementationEvaluation, EvaluationResult
from uav.evaluation.cfg import FRIEDMAN_ALPHA

class SciPyEvaluation(ImplementationEvaluation):
    def evaluate(self) -> EvaluationResult:
        measured_metric = self.measurement_data_block.measured_metric
        measurements_one = self.measurement_data_block.measurements_one
        measurements_two = self.measurement_data_block.measurements_two
        measurements_three = self.measurement_data_block.measurements_three

        friedman_p: float = -1
        wilcoxon_ps: list[float] = [-1, -1, -1]
        hommel_ps: list[float] = [-1, -1, -1]
        effect_size: float = -1

        _, friedman_p = friedmanchisquare(measurements_one, measurements_two, measurements_three)
                    
        if friedman_p < FRIEDMAN_ALPHA:
            print("Warning: Friedman p value indicates no significant difference found!")

        _, wilcoxon_ps[0] =  wilcoxon(x=measurements_one, y=measurements_two)   # type: ignore
        _, wilcoxon_ps[1] =  wilcoxon(x=measurements_two, y=measurements_three) # type: ignore
        _, wilcoxon_ps[2] =  wilcoxon(x=measurements_one, y=measurements_three) # type: ignore

        return EvaluationResult("scipy", measured_metric, friedman_p, wilcoxon_ps, hommel_ps, effect_size)

