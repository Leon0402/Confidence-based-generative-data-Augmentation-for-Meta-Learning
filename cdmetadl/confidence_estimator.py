__all__ = ["ref_set_confidence_scores", "MCDropoutConfidenceEstimation"]
from collections import defaultdict

import numpy as np

import cdmetadl.dataset
import cdmetadl.helpers.general_helpers
import cdmetadl.helpers.scoring_helpers
import cdmetadl.dataset.split

import cdmetadl.api


def caculate_confidence_on_reference_set(
    predictor: cdmetadl.api.Predictor, reference_set: cdmetadl.dataset.SetData
) -> dict:
    data, labels, _ = reference_set

    confidence_scores = defaultdict(list)
    for prediction, gt_label in zip(predictor.predict(data), labels):
        gt_label = int(gt_label)

        confidence_score = 0.0
        if np.argmax(prediction) == gt_label:
            confidence_score = np.max(prediction)

        confidence_scores[gt_label].append(confidence_score)

    return [np.mean(scores) for scores in confidence_scores.values()]


class MCDropoutConfidenceEstimation:

    def __init__(self, num_samples: int = 100, dropout_probability: float = None):
        """
        Initialize the MCDropoutConfidenceEstimation class.

        Args:
        num_samples (int): Number of samples to draw for each prediction to estimate uncertainty.
        dropout_probability (float): Probability used in dropout layers of model.
        """
        self.num_samples = num_samples
        self.dropout_probability = dropout_probability

    def estimate(self, predictor: cdmetadl.api.Predictor, reference_set: cdmetadl.dataset.SetData) -> list[float]:
        """
        Estimate the confidence of predictions using Monte Carlo Dropout.

        Args:
        predictor (cdmetadl.api.Predictor): The predictor object with dropout enabled.
        reference_set (cdmetadl.dataset.SetData): The dataset for which to estimate confidence.

        Returns:
        A list of confidence scores for each class.
        """
        for m in predictor.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                if self.dropout_probability is not None:
                    m.p = self.dropout_probability

        data, _, _ = reference_set
        class_predictions = np.array([predictor.predict(data) for _ in range(self.num_samples)])
        uncertainty_estimates = np.std(class_predictions, axis=0)

        # TODO: We measure uncertainty here, but we really want to estimate confidence (=>  confidence = 1 - uncertainty?)
        # TODO: Uncertainty values are close to each others, how can they be interpreted? How can they be converted to confidence values?
        return np.mean(uncertainty_estimates, axis=0)
