__all__ = ["MCDropoutConfidenceEstimator"]

import numpy as np

import cdmetadl.dataset
import cdmetadl.api

from .confidence_estimator import ConfidenceEstimator


class MCDropoutConfidenceEstimator(ConfidenceEstimator):
    """
    TODO General Description + class members
    """

    def __init__(self, num_samples: int = 100, dropout_probability: float = None):
        """
        Initialize the MCDropoutConfidenceEstimation class.

        Args:
            num_samples (int): Number of samples to draw for each prediction to estimate uncertainty.
            dropout_probability (float): Probability used in dropout layers of model.
        """
        self.num_samples = num_samples
        self.dropout_probability = dropout_probability

    def estimate(self, learner: cdmetadl.api.Learner,
                 data_set: cdmetadl.dataset.SetData) -> tuple[cdmetadl.dataset.SetData, list[float]]:
        """
        Generates a confidence score for each class in the given data set.

        Args:
            learner (cdmetadl.api.Learner): Learner for finetuning and confidence estimation.
            data_set (cdmetadl.dataset.SetData): Data which can be used for finetuning the learner and estimating the confidence.

        Returns:
            A tuple consisting of a dataset which can be used for finetuning the model and a list of confidence scores.
        """
        predictor = learner.fit(data_set)

        for m in predictor.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                if self.dropout_probability is not None:
                    m.p = self.dropout_probability

        class_predictions = np.array([predictor.predict(data_set.images) for _ in range(self.num_samples)])
        uncertainty_estimates = np.std(class_predictions, axis=0)

        # TODO: We measure uncertainty here, but we really want to estimate confidence (=>  confidence = 1 - uncertainty?)
        # TODO: Uncertainty values are close to each others, how can they be interpreted? How can they be converted to confidence values?
        return data_set, np.mean(uncertainty_estimates, axis=0)
