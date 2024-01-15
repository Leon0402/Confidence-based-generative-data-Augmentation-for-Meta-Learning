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

    def _init_augmentation(self, support_set: cdmetadl.dataset.SetData, conf_scores: list[float]) -> tuple:
        return None

    def _augment_class(self, cls: int, number_of_shots: int, init_args: list,
                       specific_init_args: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Augments data for a specific class using the defined image transformations. 
        Used in base class `augment` function,

        Args:
            cls (int): The class index for which the data augmentation is to be performed.
            number_of_shots (int): The number of samples to generate.
            init_args (list): General init args of the augmentation like the support_data
            specific_init_args (list): Class specific init args created in the `_init_augmentation` function, 

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the augmented data and corresponding labels for the specified class.
        """
        support_data, _, num_shots_support_set = init_args

        random_indices = np.random.randint(0, num_shots_support_set, size=number_of_shots)
        augmented_data = torch.stack([self.transform(support_data[cls][idx]) for idx in random_indices])
        augmented_labels = torch.full(size=(number_of_shots, ), fill_value=cls)
        return augmented_data, augmented_labels
