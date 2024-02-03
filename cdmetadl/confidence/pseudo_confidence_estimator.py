__all__ = ["PseudoConfidenceEstimator", "GTConfidenceEstimator"]

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
        support_set, confidence_set = cdmetadl.dataset.set_split(
            data_set, split_shot_counts=[data_set.number_of_shots - 10, 10]
        )
        predictor = learner.fit(support_set)

        confidence_scores = defaultdict(list)
        for prediction, gt_label in zip(predictor.predict(confidence_set.images).cpu().numpy(), confidence_set.labels):
            gt_label = int(gt_label)

            confidence_score = 0.0
            if np.argmax(prediction) == gt_label:
                confidence_score = np.max(prediction)

            confidence_scores[gt_label].append(confidence_score)

        return support_set, [np.mean(scores) for scores in confidence_scores.values()]


class GTConfidenceEstimator(ConfidenceEstimator):
    """
    TODO General Description + class members
    """

    def set_query_set(self, query_set: cdmetadl.dataset.SetData) -> None:
        self.query_set = query_set

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

        confidence_scores = defaultdict(list)
        for prediction, gt_label in zip(predictor.predict(self.query_set.images).cpu().numpy(), self.query_set.labels):
            gt_label = int(gt_label)

            confidence_score = 0.0
            if np.argmax(prediction) == gt_label:
                confidence_score = np.max(prediction)

            confidence_scores[gt_label].append(confidence_score)

        return data_set, [np.mean(scores) for scores in confidence_scores.values()]
