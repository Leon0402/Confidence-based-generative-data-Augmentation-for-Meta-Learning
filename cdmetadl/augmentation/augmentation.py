__all__ = ["Augmentation", "PseudoAug", "StandardAug", "GenerativeAug"]

import torch
import numpy as np
import math

import cdmetadl.dataset
import cdmetadl.helpers.general_helpers

import abc


class Augmentation(metaclass=abc.ABCMeta):

    def __init__(self, threshold: float, scale: int, keep_original_data: bool):
        self.threshold = threshold
        self.scale = scale
        self.keep_original_data = keep_original_data

    @abc.abstractmethod
    def augment(self, support_set: cdmetadl.dataset.SetData,
                conf_scores: list[float]) -> tuple[cdmetadl.dataset.SetData, list[int]]:
        pass


class PseudoAug(Augmentation):
    """
    Class for pseudo augmentation. Goes through all ways/classes, checks confidence score for particular class, calculates number of extra samples to sample for particular class as an inverse of its 
    confidence score * scale rounded down. It will randomly pick samples from the backup set and concatenate them with the original support_set. It also augments the label and original label tensors accordingly
    to account for variable shots. 

    Args:
        support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Support set used for pretraining the model on.
        backup_support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Query set for prediction after pretraining and estimation of confidence scores through softmax. 
        conf_scores: (MyLeaner) Pretrained model instatiated for meta-testing. 
        threshold: (float) Confidence value below which we want to augment the class. 
        scale: (int) Indicates how much in "fold" of the orginal set we want to add to the augmented one. 
        num_ways: (int) Number of ways in backup_support set. 
        num_shots: (int) Number of shots in backup_support set. 

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor: Augmented support set with variable shots, shots are increased accordingly for classes where confidence score was below the threshold.
        list: list of shots per ways
    """

    def __init__(
        self, augmentation_set: cdmetadl.dataset.SetData, threshold: float, scale: int, keep_original_data: bool
    ):
        super().__init__(threshold, scale, keep_original_data)
        self.augmentation_set = augmentation_set

    def augment(self, support_set: cdmetadl.dataset.SetData,
                conf_scores: list[float]) -> tuple[cdmetadl.dataset.SetData, list[int]]:
        num_ways = len(conf_scores)

        support_data, support_labels, _ = support_set
        num_shots_support_set = int(len(support_data) / num_ways)
        support_data = support_data.reshape(num_ways, num_shots_support_set, 3, 128, 128)
        support_labels = support_labels.reshape(num_ways, num_shots_support_set)

        augmentation_data, augmentation_label, _ = self.augmentation_set
        num_shots_augmentation_set = int(len(augmentation_data) / num_ways)
        augmentation_data = augmentation_data.reshape(num_ways, num_shots_augmentation_set, 3, 128, 128)
        augmentation_label = augmentation_label.reshape(num_ways, num_shots_augmentation_set)

        extended_data = []
        extended_labels = []
        shots_per_class = []
        for cls, score in enumerate(conf_scores):
            extended_data.append(support_data[cls])
            extended_labels.append(support_labels[cls])
            shots_per_class.append(num_shots_support_set)

            if score < self.threshold:
                # TODO: Check if linear interpolation makes sense. Exponential might be better, but harder to control.
                number_of_augmented_shots = math.ceil((1 - score) * num_shots_support_set * self.scale)
                if number_of_augmented_shots > num_shots_augmentation_set:
                    print(
                        f"Warning: number of augmented shots {number_of_augmented_shots} is higher than available data in augmentation set {num_shots_augmentation_set}"
                    )
                    number_of_augmented_shots = num_shots_augmentation_set

                extended_data.append(augmentation_data[cls][:number_of_augmented_shots])
                extended_labels.append(augmentation_label[cls][:number_of_augmented_shots])
                shots_per_class[-1] += number_of_augmented_shots

        return (torch.cat(extended_data, dim=0), torch.cat(extended_labels, dim=0), None), shots_per_class


class StandardAug(Augmentation):

    def augment(
        self, support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], conf_scores: list[float],
        backup_support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None
    ):
        pass


class GenerativeAug(Augmentation):

    def augment(
        self, support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], conf_scores: list[float],
        backup_support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None
    ):
        pass
