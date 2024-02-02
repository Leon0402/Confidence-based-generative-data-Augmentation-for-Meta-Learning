__all__ = ["StandardAugmentation"]

import numpy as np
import torch
import torchvision.transforms

import cdmetadl.dataset

from .augmentation import Augmentation


class StandardAugmentation(Augmentation):
    """
    StandardAugmentation is a class that extends Augmentation to provide standard image augmentation techniques.
    This class is designed to augment image datasets for machine learning tasks, especially for improving the
    performance and robustness of models in tasks like classification.

    Attributes:
        transform (torchvision.transforms.Compose): A composed series of transformations (like flip, rotation, 
                                                    and color jitter) applied to the images.
    """

    def __init__(self, augmentation_size: dict, keep_original_data: bool, device: torch.device):
        """
        Initializes the StandardAugmentation class with specified threshold, scale, and keep_original_data flags,
        along with a defined set of image transformations.

        Args:
            augmentation_size (dict): Uses for calculation how many shots should be augmented.
            keep_original_data (bool): A flag to determine whether original data should be included together with the augmented data.
        """
        super().__init__(augmentation_size, keep_original_data, device)

        # TODO: Adjust transforms, make configurable perhaps
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation((30, 60)),
            torchvision.transforms.RandomHorizontalFlip(0.2),      
            torchvision.transforms.RandomResizedCrop(128, (0.75, 0.85), antialias=True)
        ])


    def _init_augmentation(self, support_set: cdmetadl.dataset.SetData,
                           conf_scores: list[float]) -> tuple[cdmetadl.dataset.SetData, None]:
        return support_set, None

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
        augmented_data = torch.stack([
            self.transform(support_set.images_by_class[cls][idx % support_set.number_of_shots])
            for idx in range(number_of_shots)
        ])
        augmented_labels = torch.full(size=(number_of_shots, ), fill_value=cls).to(self.device)
        return augmented_data, augmented_labels
