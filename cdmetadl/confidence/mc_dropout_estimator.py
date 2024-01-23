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

       # \sum | x - \mu |

        #uncertainty_estimates = np.std(class_predictions, axis=0)
        #mean_estimates = np.mean(uncertainty_estimates, axis=0)
        
        #conf_scores = [np.exp(*(1- ue)) for ue in mean_estimates]
        #max = np.max(conf_scores)
        #min = np.min(conf_scores)
        #conf_scores = [(cs - min) / (max - min) for cs in conf_scores]


        mean_predictions = np.mean(class_predictions, axis=0)
        abs_error = np.mean([np.abs(prediction - mean_predictions) for prediction in class_predictions], axis=0)
        

        u_s = np.mean(abs_error, axis=0)
        u_s = [1 - uncert_score for uncert_score in u_s]

       # softmax_sum = np.sum([np.exp(val) for val in u_s])
       # normalized_scores = [np.exp(score)/softmax_sum for score in u_s]


        # TODO: We measure uncertainty here, but we really want to estimate confidence (=>  confidence = 1 - uncertainty?)
        # TODO: Uncertainty values are close to each others, how can they be interpreted? How can they be converted to confidence values?
        return u_s
