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

    def estimate(self, predictor: cdmetadl.api.Predictor, reference_set: cdmetadl.dataset.SetData, x_max: int, x_min: int) -> list[float]:
        """
        Estimate the confidence of predictions using Monte Carlo Dropout.
        It does num_samples time dropout passes over the query set, computes for every prediction, the mean error over these passes. 
        It then averages this across the predictions for every class, then normalizes and squashes this result into the [0, 1] interval with 
        the predefined and configurable min/max parameters. 

        Args:
        predictor (cdmetadl.api.Predictor): The predictor object with dropout enabled.
        reference_set (cdmetadl.dataset.SetData): The dataset for which to estimate confidence.
        x_max: (int) Hyperparameter that determines the cut off for the squashing of the confidence score. 
        x_main: (int) Hyperparameter that determines the cut off for the squashing of the confidence score. 

        Returns:
            A tuple consisting of a dataset which can be used for finetuning the model and a list of confidence scores.
        """
        predictor = learner.fit(data_set)

        for m in predictor.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                if self.dropout_probability is not None:
                    m.p = self.dropout_probability

        class_predictions = np.array([predictor.predict(reference_set.images) for _ in range(self.num_samples)])
        mean_predictions = np.mean(class_predictions, axis=0)
        abs_error = np.mean([np.abs(prediction - mean_predictions) for prediction in class_predictions], axis=0)
        

        uncertainty_scores = np.mean(abs_error, axis=0)
        us = [1- (np.maximum(np.minimum(score, x_max), x_min) - x_min)/(x_max - x_min) for score in uncertainty_scores]
        return us
