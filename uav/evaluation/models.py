from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np

class Metric(Enum):
    PRECISION = "Precision"
    RECALL = "Recall"
    mAP50 = "mAP50"
    mAP50_95 = "mAP50_95"

class ImplementationRegistry(Enum):
    SCIPY = "scipy"
    R = "r"
    PINGUOIN = "pinguoin"
    STAC = "stac"
    STATSMODELS_SCIPY = "statsmodels_scipy"
    STATSMODELS_R = "statsmodels_r"

class FullAnalaysisRegistry(Enum):
    R = "full_r"
    SCIPY_STATSMODELS = "full_scipy_statsmodels"
    PINGUOIN_STATSMODELS = "full_pinguoin_statsmodels"

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

@dataclass
class EffectSizeResult():
    metric: Metric
    effect_size: float

@dataclass
class MetricResult():
    experiment: str
    metric: Metric
    mean: float
    std: float

class ImplementationEvaluation(ABC):
    """Abstract class for Evaluation Implementation"""
    def __init__(self, measurement_data_block: MeasurementDataBlock, alpha: float = 0.05) -> None:
        self.measurement_data_block = measurement_data_block
        self.alpha = alpha

    @abstractmethod
    def evaluate(self) -> EvaluationResult:
        raise NotImplementedError("Please implement abstract evaluate() function for the given implementation.")

