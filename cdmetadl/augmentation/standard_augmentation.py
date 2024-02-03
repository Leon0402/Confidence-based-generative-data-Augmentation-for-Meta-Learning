__all__ = ["StandardAugmentation"]

import torch
import torchvision.transforms

import cdmetadl.dataset
import numpy as np

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

        # Transform 1
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation((30, 60)),
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomResizedCrop(128, (0.75, 0.85), antialias=True)
        ])

        # Transform 0
        # self.transform = torchvision.transforms.Compose([
        #     torchvision.transforms.RandomHorizontalFlip(),
        #     torchvision.transforms.RandomRotation(45),
        #     torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # ])

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


# class StandardAugmentation(Augmentation):

#     def __init__(
#         self, augmentation_size: dict, keep_original_data: bool, device: torch.device, severity=3, width=3, depth=-1,
#         alpha=1.0
#     ):
#         super().__init__(augmentation_size, keep_original_data, device)
#         self.severity = severity
#         self.width = width
#         self.depth = depth if depth > 0 else np.random.randint(1, 4)
#         self.alpha = alpha
#         self.augmentations = [
#             torchvision.transforms.ColorJitter(
#                 brightness=0.2 * severity, contrast=0.2 * severity, saturation=0.2 * severity, hue=0.1 * severity
#             ),
#             torchvision.transforms.RandomAffine(
#                 degrees=20 * severity, translate=(0.1 * severity, 0.1 * severity), scale=(0.9, 1.1), shear=10 * severity
#             ),
#             torchvision.transforms.RandomGrayscale(p=0.2 * severity),
#             # Add more diverse and task-specific augmentations as needed
#         ]

#     def _init_augmentation(self, support_set: cdmetadl.dataset.SetData,
#                            conf_scores: list[float]) -> tuple[cdmetadl.dataset.SetData, None]:
#         return support_set, None

#     def augmix(self, image):
#         """Apply AugMix augmentations to a single image."""
#         mixed = torch.zeros_like(image)
#         weights = np.random.dirichlet([self.alpha] * self.width)

#         for i in range(self.width):
#             aug_image = image
#             depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
#             for _ in range(depth):
#                 op = np.random.choice(self.augmentations)
#                 aug_image = op(aug_image)
#             mixed += weights[i] * aug_image

#         # Mix the original image
#         mixed = (mixed + image) / 2
#         return mixed

#     def _augment_class(self, cls: int, support_set: cdmetadl.dataset.SetData, number_of_shots: int,
#                        init_args: list) -> tuple[torch.Tensor, torch.Tensor]:
#         augmented_data = []
#         for idx in range(number_of_shots):
#             image = support_set.images_by_class[cls][idx % support_set.number_of_shots]
#             image = self.augmix(image)
#             augmented_data.append(image)

#         augmented_data = torch.stack(augmented_data)
#         augmented_labels = torch.full(size=(number_of_shots, ), fill_value=cls).to(self.device)
#         return augmented_data, augmented_labels
