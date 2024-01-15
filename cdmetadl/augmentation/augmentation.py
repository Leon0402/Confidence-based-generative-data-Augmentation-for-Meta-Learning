__all__ = ["Augmentation"]

import abc
import math

import torch

import cdmetadl.dataset
import cdmetadl.helpers.general_helpers


class Augmentation(metaclass=abc.ABCMeta):

    def __init__(self, threshold: float, scale: int, keep_original_data: bool):
        """
        Initialize the Augmentation class.

        :param threshold: Threshold for determining the amount of augmentation.
        :param scale: Scale factor for deciding how much data to generate per class.
        :param keep_original_data: Flag to keep the original data alongside the augmented data.
        """
        self.threshold = threshold
        self.scale = scale
        self.keep_original_data = keep_original_data

    def augment(self, support_set: cdmetadl.dataset.SetData,
                conf_scores: list[float]) -> tuple[cdmetadl.dataset.SetData, list[int]]:
        """
        Perform augmentation on the support set based on confidence scores.

        :param support_set: The support set to augment.
        :param conf_scores: List of confidence scores corresponding to each class.
        :return: A tuple of the augmented dataset and the number of shots per class.
        """
        num_ways = len(conf_scores)
        support_data, support_labels, _ = support_set
        num_shots_support_set = int(len(support_data) / num_ways)
        support_data = support_data.reshape(num_ways, num_shots_support_set, 3, 128, 128)
        support_labels = support_labels.reshape(num_ways, num_shots_support_set)

        init_args = (support_data, support_labels, num_shots_support_set)
        specific_init_args = self._init_augmentation(support_set, conf_scores)

        extended_data, extended_labels, shots_per_class = self._augment(conf_scores, init_args, specific_init_args)
        return (torch.cat(extended_data, dim=0), torch.cat(extended_labels, dim=0), None), shots_per_class

    def _augment(self, conf_scores: list[float], init_args: tuple,
                 specific_init_args: tuple) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Internal method to handle the augmentation logic.

        :param conf_scores: List of confidence scores.
        :param init_args: Initial arguments including support data and labels.
        :param specific_init_args: Specific arguments for the augmentation.
        :return: Augmented data and labels, and shots per class.
        """
        support_data, support_labels, num_shots_support_set = init_args

        extended_data = []
        extended_labels = []
        shots_per_class = []
        for cls, score in enumerate(conf_scores):
            shots_per_class.append(0)

            if self.keep_original_data:
                extended_data.append(support_data[cls])
                extended_labels.append(support_labels[cls])
                shots_per_class[-1] *= num_shots_support_set

            if score < self.threshold:
                # TODO: Check if linear interpolation makes sense. Exponential might be better, but harder to control.
                number_of_augmented_shots = math.ceil((1 - score) * num_shots_support_set * self.scale)
                data, labels = self._augment_class(cls, number_of_augmented_shots, init_args, specific_init_args)
                extended_data.append(data)
                extended_labels.append(labels)
                shots_per_class[-1] += len(labels)

        return extended_data, extended_labels, shots_per_class

    @abc.abstractmethod
    def _init_augmentation(self, support_set: cdmetadl.dataset.SetData, conf_scores: list[float]) -> tuple:
        """
        Abstract method to initialize augmentation-specific parameters.

        :param support_set: The support set.
        :param conf_scores: Confidence scores for each class.
        :return: Specific initialization arguments for augmentation.
        """
        pass

    @abc.abstractmethod
    def _augment_class(self, cls: int, number_of_shots: int, init_args: list,
                       specific_init_args: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Abstract method to perform augmentation for a specific class.

        :param cls: Class index to augment.
        :param number_of_shots: Number of augmented shots to generate.
        :param init_args: Initial arguments including support data and labels.
        :param specific_init_args: Specific arguments for the augmentation.
        :return: tuple of the augmented data and labels for the specified class.
        """
        pass
