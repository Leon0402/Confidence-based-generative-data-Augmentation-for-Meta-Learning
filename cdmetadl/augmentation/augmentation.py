__all__ = ["Augmentation", "PseudoAug", "StandardAug", "GenerativeAug"]

import torch
import math
import numpy as np
import torchvision.transforms as transforms

import cdmetadl.dataset
import cdmetadl.helpers.general_helpers

import abc


class Augmentation(metaclass=abc.ABCMeta):

    def __init__(self, threshold: float, scale: int, keep_original_data: bool):
        self.threshold = threshold
        self.scale = scale
        self.keep_original_data = keep_original_data

    def augment(self, support_set: cdmetadl.dataset.SetData,
                conf_scores: list[float]) -> tuple[cdmetadl.dataset.SetData, list[int]]:

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
                data, labels, shots = self._augment_class(cls, number_of_augmented_shots, init_args, specific_init_args)
                extended_data.append(data)
                extended_labels.append(labels)
                shots_per_class[-1] += shots

        return extended_data, extended_labels, shots_per_class

    def _init_augmentation(self, support_set: cdmetadl.dataset.SetData, conf_scores: list[float]) -> tuple:
        return None

    @abc.abstractmethod
    def _augment_class(self, cls: int, number_of_shots: int, init_args: list,
                       specific_init_args: list) -> tuple[torch.Tensor, torch.Tensor, int]:
        pass


class PseudoAug(Augmentation):

    def __init__(
        self, augmentation_set: cdmetadl.dataset.SetData, threshold: float, scale: int, keep_original_data: bool
    ):
        super().__init__(threshold, scale, keep_original_data)
        self.augmentation_set = augmentation_set

    def _init_augmentation(self, support_set: cdmetadl.dataset.SetData, conf_scores: list[float]) -> tuple:
        num_ways = len(conf_scores)
        augmentation_data, augmentation_label, _ = self.augmentation_set
        num_shots_augmentation_set = int(len(augmentation_data) / num_ways)
        augmentation_data = augmentation_data.reshape(num_ways, num_shots_augmentation_set, 3, 128, 128)
        augmentation_label = augmentation_label.reshape(num_ways, num_shots_augmentation_set)

        return augmentation_data, augmentation_label, num_shots_augmentation_set

    def _augment_class(self, cls: int, number_of_shots: int, init_args: list,
                       specific_init_args: list) -> tuple[list, list, int]:
        augmentation_data, augmentation_label, num_shots_augmentation_set = specific_init_args

        if number_of_shots > num_shots_augmentation_set:
            print(
                f"Warning: number of augmented shots {number_of_shots} is higher than available data in augmentation set {num_shots_augmentation_set}"
            )
            number_of_shots = num_shots_augmentation_set

        return augmentation_data[cls][:number_of_shots], augmentation_label[cls][:number_of_shots], number_of_shots


class StandardAug(Augmentation):

    def __init__(self, threshold: float, scale: int, keep_original_data: bool):
        super().__init__(threshold, scale, keep_original_data)

        # TODO: Adjust which transforms to use (perhaps ColorJitter)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            transforms.ToTensor()
        ])

    def _augment_class(self, cls: int, number_of_shots: int, init_args: list,
                       specific_init_args: list) -> tuple[list, list, int]:
        support_data, _, num_shots_support_set = init_args

        extended_data = []
        for _ in range(number_of_shots):
            random_image_index = np.random.randint(0, num_shots_support_set)
            transformed_image = self.transform(support_data[cls][random_image_index])
            extended_data.append(transformed_image)

        return torch.stack(extended_data), torch.tensor([cls] * number_of_shots), number_of_shots


class GenerativeAug(Augmentation):

    def augment(self, support_set: cdmetadl.dataset.SetData,
                conf_scores: list[float]) -> tuple[cdmetadl.dataset.SetData, list[int]]:
        pass
