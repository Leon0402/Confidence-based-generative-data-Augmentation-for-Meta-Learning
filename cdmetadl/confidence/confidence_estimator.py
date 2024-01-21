__all__ = ["ConfidenceEstimator"]

import abc

import cdmetadl.dataset


class ConfidenceEstimator(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def estimate(self, predictor: cdmetadl.api.Predictor, reference_set: cdmetadl.dataset.SetData) -> list[float]:
        """
        Generates a confidence score for each class in the given reference set.

        Args:
        predictor (cdmetadl.api.Predictor): The predictor for which to estimate confidence.
        reference_set (cdmetadl.dataset.SetData): The dataset for which to estimate confidence.

        Returns:
        A list of confidence scores for each class.
        """
        pass
