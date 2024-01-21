__all__ = ["MCDropoutConfidenceEstimator"]

import numpy as np

import cdmetadl.dataset
import cdmetadl.api

from .confidence_estimator import ConfidenceEstimator


class MCDropoutConfidenceEstimator(ConfidenceEstimator):

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

        class_predictions = np.array([predictor.predict(reference_set.images) for _ in range(self.num_samples)])
        uncertainty_estimates = np.std(class_predictions, axis=0)

        # TODO: We measure uncertainty here, but we really want to estimate confidence (=>  confidence = 1 - uncertainty?)
        # TODO: Uncertainty values are close to each others, how can they be interpreted? How can they be converted to confidence values?
        return np.mean(uncertainty_estimates, axis=0)
