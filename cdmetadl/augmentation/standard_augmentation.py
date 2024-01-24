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

    def __init__(self, threshold: float, scale: int, keep_original_data: bool):
        """
        Initializes the StandardAugmentation class with specified threshold, scale, and keep_original_data flags,
        along with a defined set of image transformations.

        Args:
            threshold (float): A threshold value for deciding which classes to augment.
            scale (int): A scale factor for deciding how many samples per classes should be created.
            keep_original_data (bool): A flag to determine whether original data should be included together with the augmented data.
        """
        super().__init__(threshold, scale, keep_original_data)

        # TODO: Adjust transforms, make configurable perhaps
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(45),
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
        random_indices = np.random.randint(0, support_set.number_of_shots, size=number_of_shots)
        augmented_data = torch.stack([self.transform(support_set.images_by_class[cls][idx]) for idx in random_indices])
        augmented_labels = torch.full(size=(number_of_shots, ), fill_value=cls)
        return augmented_data, augmented_labels
