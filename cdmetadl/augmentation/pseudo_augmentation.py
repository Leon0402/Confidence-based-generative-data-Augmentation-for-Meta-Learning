__all__ = ["PseudoAugmentation"]

import torch

import cdmetadl.dataset

from .augmentation import Augmentation


class PseudoAugmentation(Augmentation):
    """
    PseudoAugmentation has a second dataset where it takes real examples from when asked to augment data.
    """

    def __init__(
        self, augmentation_set: cdmetadl.dataset.SetData, threshold: float, scale: int, keep_original_data: bool
    ):
        """
        Initialize the PseudoAugmentation class.

        :param augmentation_set: The dataset where the data for augmentation is taken from
        :param threshold: Threshold for determining the amount of augmentation.
        :param scale: Scale factor for deciding how much data to generate per class.
        :param keep_original_data: Flag to keep the original data alongside the augmented data.
        """
        super().__init__(threshold, scale, keep_original_data)
        self.augmentation_set = augmentation_set

    def _init_augmentation(self, support_set: cdmetadl.dataset.SetData, conf_scores: list[float]) -> tuple:
        augmentation_data, augmentation_label, _ = self.augmentation_set

        num_ways = len(conf_scores)
        num_shots = int(len(augmentation_data) / num_ways)

        augmentation_data = augmentation_data.reshape(num_ways, num_shots, 3, 128, 128)
        augmentation_label = augmentation_label.reshape(num_ways, num_shots)

        return augmentation_data, augmentation_label, num_shots

    def _augment_class(self, cls: int, number_of_shots: int, init_args: list,
                       specific_init_args: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform augmentation for a specific class.

        :param cls: Class index to augment.
        :param number_of_shots: Number of augmented shots to generate.
        :param init_args: Initial arguments including support data and labels.
        :param specific_init_args: Specific arguments for the augmentation.
        :return: tuple of the augmented data and labels for the specified class.
        """
        augmentation_data, augmentation_label, num_shots_augmentation_set = specific_init_args

        if number_of_shots > num_shots_augmentation_set:
            print(
                f"Warning: number of augmented shots {number_of_shots} is higher than available data in augmentation set {num_shots_augmentation_set}"
            )
            number_of_shots = num_shots_augmentation_set

        return augmentation_data[cls][:number_of_shots], augmentation_label[cls][:number_of_shots]
