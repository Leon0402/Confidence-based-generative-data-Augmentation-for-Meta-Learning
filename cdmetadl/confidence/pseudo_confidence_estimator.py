__all__ = ["PseudoConfidenceEstimator"]

from collections import defaultdict

import numpy as np

import cdmetadl.dataset
import cdmetadl.api

from .confidence_estimator import ConfidenceEstimator


class PseudoConfidenceEstimator(ConfidenceEstimator):

    def estimate(self, predictor: cdmetadl.api.Predictor, reference_set: cdmetadl.dataset.SetData) -> list[float]:
        """
        Estimate the confidence of predictions using Monte Carlo Dropout.

        Args:
        predictor (cdmetadl.api.Predictor): The predictor object with dropout enabled.
        reference_set (cdmetadl.dataset.SetData): The dataset for which to estimate confidence.

        Returns:
        A list of confidence scores for each class.
        """
        confidence_scores = defaultdict(list)
        for prediction, gt_label in zip(predictor.predict(reference_set.images), reference_set.labels):
            gt_label = int(gt_label)

            confidence_score = 0.0
            if np.argmax(prediction) == gt_label:
                confidence_score = np.max(prediction)

            confidence_scores[gt_label].append(confidence_score)

        return [np.mean(scores) for scores in confidence_scores.values()]
