import json

import torch
from jaxtyping import Float

from .logging_setup import create_logger
from .phi_3_5_probe import ProbesForScenario

import numpy as np
from dataclasses import dataclass

from .utils import is_binary, bear_jax_typed

logger = create_logger(__name__)


@dataclass
class TraditionalMetrics:
    """
    Holds metrics derived from the traditional (hard) confusion matrix.
    All metrics are based on binary predictions after thresholding.
    """
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass
class SoftMetrics:
    """
    Holds metrics derived from the soft confusion matrix.
    All metrics incorporate prediction confidences rather than binary decisions.
    """
    soft_precision: float
    soft_recall: float
    soft_f1: float
    # average level of confidence in the correct answer (across all cases)
    avg_confidence_correct: float
    # average level of confidence in the incorrect answer (across all cases)
    avg_confidence_incorrect: float


@dataclass
class ConfusionMetricsState:
    """Serializable state of ConfusionMetrics"""
    threshold: float
    tp_count: int
    fp_count: int
    tn_count: int
    fn_count: int
    tp_sum: float
    fp_sum: float
    tn_sum: float
    fn_sum: float
    squared_error_sum: float
    n_samples: int


class ConfusionMetrics:
    """
    A comprehensive binary classifier evaluation:
    1. Traditional confusion matrix (using thresholded predictions)
    2. Soft confusion matrix (using prediction confidences)
    3. Brier score (a 'proper' scoring rule)
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize the metrics tracker.

        Args:
            :param threshold: Decision threshold for converting probabilities to binary predictions
                      in the traditional confusion matrix
        """
        self.threshold = threshold
        # Traditional confusion matrix counts
        self.tp_count: int = 0
        self.fp_count: int = 0
        self.tn_count: int = 0
        self.fn_count: int = 0

        # Soft confusion matrix sums
        self.tp_sum: float = 0.0
        self.fp_sum: float = 0.0
        self.tn_sum: float = 0.0
        self.fn_sum: float = 0.0

        # Components for Brier score calculation
        self.squared_error_sum: float = 0.0
        self.n_samples: int = 0

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.tp_count = self.fp_count = self.tn_count = self.fn_count = self.n_samples = 0
        self.tp_sum = self.fp_sum = self.tn_sum = self.fn_sum = self.squared_error_sum = 0.0

    def update(self, y_label: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """
        Update metrics with a batch of predictions.

        Args:
            y_label: Array of labels (0 or 1) for the data records
            y_pred_proba: Array of predicted probabilities for class 1 for the data records
        """
        if not (0 <= y_pred_proba).all() and (y_pred_proba <= 1).all():
            logger.error("Predicted probabilities outside [0,1] range")
            raise ValueError("Predicted probabilities outside [0,1] range")

        # Convert to binary predictions for traditional confusion matrix
        y_pred_binary = (y_pred_proba >= self.threshold).astype(int)

        # Update traditional confusion matrix
        self.tp_count += int(np.sum((y_label == 1) & (y_pred_binary == 1)))
        self.fp_count += int(np.sum((y_label == 0) & (y_pred_binary == 1)))
        self.tn_count += int(np.sum((y_label == 0) & (y_pred_binary == 0)))
        self.fn_count += int(np.sum((y_label == 1) & (y_pred_binary == 0)))

        # Update soft confusion matrix
        actual_positive_records_mask = y_label == 1
        actual_negative_records_mask = y_label == 0

        self.tp_sum += float(np.sum(y_pred_proba[actual_positive_records_mask]))
        self.fn_sum += float(np.sum((1 - y_pred_proba)[actual_positive_records_mask]))
        self.fp_sum += float(np.sum(y_pred_proba[actual_negative_records_mask]))
        self.tn_sum += float(np.sum((1 - y_pred_proba)[actual_negative_records_mask]))

        # Update Brier score components
        self.squared_error_sum += float(np.sum((y_pred_proba - y_label) ** 2))
        self.n_samples += len(y_label)

    def get_traditional_confusion_matrix(self) -> np.ndarray:
        return np.array([
            [self.tp_count, self.fp_count],
            [self.fn_count, self.tn_count]
        ])

    def get_soft_confusion_matrix(self) -> np.ndarray:
        """Returns the confidence-weighted confusion matrix."""
        return np.array([
            [self.tp_sum, self.fp_sum],
            [self.fn_sum, self.tn_sum]
        ])

    def get_brier_score(self) -> float:
        """
        Calculate the Brier score.

        The Brier score is a proper scoring rule that measures the mean squared error
        of probabilistic predictions. Lower values indicate better performance.
        """
        return self.squared_error_sum / self.n_samples if self.n_samples > 0 else np.nan

    def get_traditional_metrics(self) -> TraditionalMetrics:
        total = self.tp_count + self.tn_count + self.fp_count + self.fn_count
        if total == 0:
            return TraditionalMetrics(accuracy=np.nan, precision=np.nan, recall=np.nan, f1=np.nan)

        precision_denom = self.tp_count + self.fp_count
        recall_denom = self.tp_count + self.fn_count

        precision = self.tp_count / precision_denom if precision_denom > 0 else np.nan
        recall = self.tp_count / recall_denom if recall_denom > 0 else np.nan

        return TraditionalMetrics(
            accuracy=(self.tp_count + self.tn_count) / total, precision=precision, recall=recall,
            f1=2 * (precision * recall) / (precision + recall) if precision + recall > 0 else np.nan
        )

    def get_soft_metrics(self) -> SoftMetrics:
        precision_denom = self.tp_sum + self.fp_sum
        recall_denom = self.tp_sum + self.fn_sum

        precision = self.tp_sum / precision_denom if precision_denom > 0 else np.nan
        recall = self.tp_sum / recall_denom if recall_denom > 0 else np.nan

        return SoftMetrics(
            soft_precision=precision, soft_recall=recall,
            soft_f1=2 * (precision * recall) / (precision + recall) if precision + recall > 0 else np.nan,
            avg_confidence_correct=(self.tp_sum + self.tn_sum) / self.n_samples if self.n_samples > 0 else np.nan,
            avg_confidence_incorrect=(self.fp_sum + self.fn_sum) / self.n_samples if self.n_samples > 0 else np.nan
        )

    def to_dict(self) -> dict:
        return ConfusionMetricsState(
            threshold=self.threshold, tp_count=self.tp_count, fp_count=self.fp_count, tn_count=self.tn_count,
            fn_count=self.fn_count, tp_sum=self.tp_sum, fp_sum=self.fp_sum, tn_sum=self.tn_sum, fn_sum=self.fn_sum,
            squared_error_sum=self.squared_error_sum, n_samples=self.n_samples
        ).__dict__

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> 'ConfusionMetrics':
        metrics = cls(threshold=data['threshold'])
        state = ConfusionMetricsState(**data)
        metrics.__dict__.update(state.__dict__)
        return metrics

    @classmethod
    def from_json(cls, json_str: str) -> 'ConfusionMetrics':
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def combine(cls, *dsets_metrics: 'ConfusionMetrics') -> 'ConfusionMetrics':
        """
        Combine multiple ConfusionMetrics instances for various smaller datasets into a single instance for the
        classifier's performance across all of those datasets.
        """
        combined = cls()
        if dsets_metrics:
            combined.threshold = dsets_metrics[0].threshold
        else:
            logger.warning("No metrics objects to combine")

        for one_dset_metrics in dsets_metrics:
            assert one_dset_metrics.threshold == combined.threshold, "Cannot combine metrics with different thresholds"
            combined.tp_count += one_dset_metrics.tp_count
            combined.fp_count += one_dset_metrics.fp_count
            combined.tn_count += one_dset_metrics.tn_count
            combined.fn_count += one_dset_metrics.fn_count
            combined.tp_sum += one_dset_metrics.tp_sum
            combined.fp_sum += one_dset_metrics.fp_sum
            combined.tn_sum += one_dset_metrics.tn_sum
            combined.fn_sum += one_dset_metrics.fn_sum
            combined.squared_error_sum += one_dset_metrics.squared_error_sum
            combined.n_samples += one_dset_metrics.n_samples
        return combined


@dataclass
class MetricsForDatasetProbes:
    """
    Holds metrics for a dataset, computed from the probes that were trained on that dataset using different layers of
    activations.
    """
    lyr18_probe_metrics: ConfusionMetrics
    lyr18_baseline_linear_probe_metrics: ConfusionMetrics

    def to_dict(self) -> dict:
        """Serialize metrics to dictionary"""
        return {
            "lyr18_probe_metrics": self.lyr18_probe_metrics.to_dict(),
            "lyr18_baseline_linear_probe_metrics": self.lyr18_baseline_linear_probe_metrics.to_dict()
        }

    def to_json(self) -> str:
        """Serialize metrics to JSON string"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> 'MetricsForDatasetProbes':
        """Create MetricsForDatasetProbes instance from dictionary"""
        return cls(
            lyr18_probe_metrics=ConfusionMetrics.from_dict(data['lyr18_probe_metrics']),
            lyr18_baseline_linear_probe_metrics=ConfusionMetrics.from_dict(data['lyr18_baseline_linear_probe_metrics'])
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'MetricsForDatasetProbes':
        """Create MetricsForDatasetProbes instance from JSON string"""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def combine(cls, *dsets_metrics: 'MetricsForDatasetProbes') -> 'MetricsForDatasetProbes':
        """
        Combine multiple MetricsForDatasetProbes instances for various smaller datasets into a single instance for the
        classifier's performance across all of those datasets.
        """
        combined = cls(
            lyr18_probe_metrics=ConfusionMetrics.combine(*[m.lyr18_probe_metrics for m in dsets_metrics]),
            lyr18_baseline_linear_probe_metrics=
            ConfusionMetrics.combine(*[m.lyr18_baseline_linear_probe_metrics for m in dsets_metrics])
        )
        return combined


@bear_jax_typed
def evaluate_classifier_performance(probes: ProbesForScenario, activations: Float[torch.Tensor, "n_recs act_sz"],
                                    truth_labels: Float[torch.Tensor, "n_recs 1"], threshold=0.5
                                    ) -> MetricsForDatasetProbes:
    """
    Evaluate the performance of a classifier on some segment of some dataset

    Args:
        :param probes: Probes trained on activations of the target SLM at different layers
        :param activations: activations at multiple layers for the dataset (possibly only a segment of the dataset)
        :param truth_labels: true labels for the dataset (possibly only a segment of the dataset)
        :param threshold: Decision threshold for converting probabilities to binary predictions
    Returns:
        MetricsForDatasetProbes: Metrics for the classifier's performance on (the given segment of) the dataset

    """
    assert is_binary(truth_labels)
    assert activations.shape[1] == probes.ttpd_probe.activation_size == probes.baseline_linear_probe.activation_size

    lyr18_probe_metrics = ConfusionMetrics(threshold)
    lyr18_baseline_linear_probe_metrics = ConfusionMetrics(threshold)

    lyr18_probe_preds = probes.ttpd_probe(activations).detach()
    lyr18_baseline_linear_probe_preds = probes.baseline_linear_probe(activations).detach()

    labels_np = truth_labels.numpy()
    lyr18_probe_metrics.update(labels_np, lyr18_probe_preds.numpy())
    lyr18_baseline_linear_probe_metrics.update(labels_np, lyr18_baseline_linear_probe_preds.numpy())

    return MetricsForDatasetProbes(
        lyr18_probe_metrics=lyr18_probe_metrics, lyr18_baseline_linear_probe_metrics=lyr18_baseline_linear_probe_metrics
    )
