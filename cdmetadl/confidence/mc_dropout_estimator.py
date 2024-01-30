__all__ = ["MCDropoutConfidenceEstimator"]

import numpy as np

import cdmetadl.dataset
import cdmetadl.api

from .confidence_estimator import ConfidenceEstimator


class MCDropoutConfidenceEstimator(ConfidenceEstimator):
    """
    TODO General Description + class members
    """

    def __init__(
        self, num_samples: int = 100, dropout_probability: float = None, x_min: float = 0.01, x_max: float = 0.05
    ):
        """
        Initialize the MCDropoutConfidenceEstimation class.

        Args:
            num_samples (int): Number of samples to draw for each prediction to estimate uncertainty.
            dropout_probability (float): Probability used in dropout layers of model.
        """
        self.num_samples = num_samples
        self.dropout_probability = dropout_probability
        self.x_min = x_min
        self.x_max = x_max

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

        # TODO: Just for testing
        for m in reversed(list(predictor.model.modules())):
            if m.__class__.__name__.startswith('Dropout'):
                m.p = 0
                break

        class_predictions = np.array([
            predictor.predict(data_set.images).cpu().numpy() for _ in range(self.num_samples)
        ])

        mean_predictions = np.mean(class_predictions, axis=0)
        abs_error = np.mean([np.abs(prediction - mean_predictions) for prediction in class_predictions], axis=0)

        uncertainty_scores = np.mean(abs_error, axis=0)
        us = [
            1 - (np.maximum(np.minimum(score, self.x_max), self.x_min) - self.x_min) / (self.x_max - self.x_min)
            for score in uncertainty_scores
        ]
        return data_set, us
