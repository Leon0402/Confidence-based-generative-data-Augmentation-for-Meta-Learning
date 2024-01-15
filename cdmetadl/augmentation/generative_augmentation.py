__all__ = ["GenerativeAugmentation"]

from typing import override

import torch

import cdmetadl.dataset

from .augmentation import Augmentation


class GenerativeAugmentation(Augmentation):

    @override
    def _init_augmentation(self, support_set: cdmetadl.dataset.SetData, conf_scores: list[float]) -> tuple:
        return None

    @override
    def _augment_class(self, cls: int, number_of_shots: int, init_args: list,
                       specific_init_args: list) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()
