__all__ = ["PseudoAugmentation"]

import torch

import cdmetadl.dataset

from .augmentation import Augmentation


class PseudoAugmentation(Augmentation):
    """
    PseudoAugmentation has a second dataset where it takes real examples from when asked to augment data.
    """

    def __init__(self, augmentation_size: dict, keep_original_data: bool, device: torch.device):
        """
        Initialize the PseudoAugmentation class.

        :param augmentation_size: Uses for calculation how many shots should be augmented.
        :param keep_original_data: Flag to keep the original data alongside the augmented data.
        """
        super().__init__(augmentation_size, keep_original_data, device)

    def _init_augmentation(self, support_set: cdmetadl.dataset.SetData,
                           conf_scores: list[float]) -> tuple[cdmetadl.dataset.SetData, cdmetadl.dataset.SetData]:
        """
        Initialize augmentation-specific parameters.

        :param support_set: The support set.
        :param conf_scores: Confidence scores for each class.
        :return: Specific initialization arguments for augmentation.
        """
        return cdmetadl.dataset.set_split(support_set, split_shot_counts=[support_set.number_of_shots - 10, 10])

    def _augment_class(self, cls: int, support_set: cdmetadl.dataset.SetData, number_of_shots: int,
                       init_args: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Augments data for a specific class using the defined image transformations. 
        Used in base class `augment` function.

        :param cls: Class index to augment.
        :param support_set: The support set to augment.
        :param number_of_shots: Number of augmented shots to generate.
        :param init_args: Arguments returned by the `_init_augmentation` function. 
        :return: tuple of the augmented data and labels for the specified class.
        """
        augmentation_set: cdmetadl.dataset.SetData = init_args

        if number_of_shots > augmentation_set.number_of_shots:
            print(
                f"Warning: number of augmented shots {number_of_shots} is higher than available data in augmentation set {augmentation_set.number_of_shots}"
            )
            number_of_shots = augmentation_set.number_of_shots

        return augmentation_set.images_by_class[cls][:number_of_shots], augmentation_set.labels_by_class[
            cls][:number_of_shots]
