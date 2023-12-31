__all__ = ["PseudoConfidenceEstimator"]

from collections import defaultdict

import numpy as np

import cdmetadl.dataset
import cdmetadl.api

from .confidence_estimator import ConfidenceEstimator


class PseudoConfidenceEstimator(ConfidenceEstimator):
    """
    TODO General Description + class members
    """

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
        # TODO: Use split based on probability (1/3, 2/3)
        support_set, confidence_set = cdmetadl.dataset.set_split(data_set, number_of_splits=2)
        predictor = learner.fit(support_set)

        confidence_scores = defaultdict(list)
        for prediction, gt_label in zip(predictor.predict(confidence_set.images), confidence_set.labels):
            gt_label = int(gt_label)

            confidence_score = 0.0
            if np.argmax(prediction) == gt_label:
                confidence_score = np.max(prediction)

            confidence_scores[gt_label].append(confidence_score)

        return support_set, [np.mean(scores) for scores in confidence_scores.values()]
