__all__ = ["Augmentation"]

import abc
import math
from typing import Any

import torch
import numpy as np
from tqdm import tqdm
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

    def augment(self, support_set: cdmetadl.dataset.SetData, conf_scores: list[float]) -> cdmetadl.dataset.SetData:
        """
        Perform augmentation on the support set based on confidence scores.

        :param support_set: The support set to augment.
        :param conf_scores: List of confidence scores corresponding to each class.
        :return: The augmented dataset.
        """
        support_set, init_args = self._init_augmentation(support_set, conf_scores)

        extended_data = []
        extended_labels = []
        shots_per_class = []
        for cls, score in tqdm(enumerate(conf_scores), total=len(conf_scores), leave=False, desc=f"Augmenting class", unit=""):
            shots_per_class.append(0)

            if self.keep_original_data:
                extended_data.append(support_set.images_by_class[cls])
                extended_labels.append(support_set.labels_by_class[cls])
                shots_per_class[-1] += support_set.number_of_shots

            if score < self.threshold:
                # TODO: Check if linear interpolation makes sense. Exponential might be better, but harder to control.
                number_of_augmented_shots = math.ceil((1 - score) * support_set.number_of_shots * self.scale)
                data, labels = self._augment_class(cls, support_set, number_of_augmented_shots, init_args)
                extended_data.append(data)
                extended_labels.append(labels)
                shots_per_class[-1] += len(labels)

        return cdmetadl.dataset.SetData(
            images=torch.cat(extended_data, dim=0),
            labels=torch.cat(extended_labels, dim=0),
            number_of_ways=support_set.number_of_ways,
            number_of_shots=np.array(shots_per_class),
            class_names=support_set.class_names,
        )

    @abc.abstractmethod
    def _init_augmentation(self, support_set: cdmetadl.dataset.SetData,
                           conf_scores: list[float]) -> tuple[cdmetadl.dataset.SetData, Any]:
        """
        Abstract method to initialize augmentation-specific parameters.

        :param support_set: The support set.
        :param conf_scores: Confidence scores for each class.
        :return: Specific initialization arguments for augmentation.
        """
        pass

    @abc.abstractmethod
    def _augment_class(self, cls: int, support_set: cdmetadl.dataset.SetData, number_of_shots: int,
                       init_args: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Abstract method to perform augmentation for a specific class.

        :param cls: Class index to augment.
        :param support_set: The support set to augment.
        :param number_of_shots: Number of augmented shots to generate.
        :param init_args: Arguments returned by the `_init_augmentation` function. 
        :return: tuple of the augmented data and labels for the specified class.
        """
        pass
