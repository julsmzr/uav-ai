from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np

class Metric(Enum):
    PRECISION = "P"
    RECALL = "R"
    mAP50 = "mAP50"
    mAP50_95 = "mAP50-95"

@dataclass
class MeasurementDataBlock():
    """Single data block of a stream of metric results."""
    measured_metric: Metric
    measurements_one: np.ndarray
    measurements_two: np.ndarray
    measurements_three: np.ndarray

@dataclass
class EvaluationResult():
    """Evaluation Results of an implementation for statistical analysis."""
    implementation: str
    measured_metric: Metric
    friedman_p: float
    wilcoxon_ps: list[float]
    hommel_ps: list[float]
    effect_size: float

class ImplementationEvaluation(ABC):
    def __init__(self, measurement_data_block: MeasurementDataBlock) -> None:
        self.measurement_data_block = measurement_data_block

    @abstractmethod
    def evaluate(self) -> EvaluationResult:
        raise NotImplementedError("Please implement abstract evaluate() function for the given implementation.")