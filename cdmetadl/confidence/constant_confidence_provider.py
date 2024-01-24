__all__ = ["ConstantConfidenceProvider"]

import abc

import cdmetadl.dataset
import cdmetadl.api


class ConstantConfidenceProvider():
    """
    TODO: Description
    """

    def __init__(self, confidence: float):
        self.confidence = confidence

    @abc.abstractmethod
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
        return data_set, [self.confidence] * data_set.number_of_ways
