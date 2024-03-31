__all__ = ["Task", "SetData"]

from dataclasses import dataclass
from functools import cached_property

import torch
import numpy as np


@dataclass()
class SetData:
    """
    Image tensor of shape [number_of_ways x number_of_ways[way], 3, 128, 128].
    """
    images: torch.Tensor
    """
    Labels Tensor of shape [number_of_ways x numer_of_shots[way]].
    """
    labels: torch.Tensor
    """
    Number of ways (classes).
    """
    number_of_ways: int
    """
    Number of shots per class.
    """
    number_of_shots: np.ndarray[int] | int
    """
    Original class names used in the task.
    """
    class_names: list[str]

    @cached_property
    def images_by_class(self) -> torch.Tensor:
        """
        Unsequeezes first dimension of images into two, thus the shape is [number_of_ways, number_of_shots, 3, 128, 128],
        """
        if type(self.number_of_shots) is not int:
            raise ValueError(f"Cannot unseqeeze as number of shots per class varies")

        return self.images.reshape(self.number_of_ways, self.number_of_shots, 3, 128, 128)

    @cached_property
    def labels_by_class(self) -> torch.Tensor:
        """
        Unsequeezes first dimension of labels into two, thus the shape is [number_of_ways, number_of_shots],
        """
        if type(self.number_of_shots) is not int:
            raise ValueError(f"Cannot unseqeeze as number of shots per class varies")

        return self.labels.reshape(self.number_of_ways, self.number_of_shots)

    @cached_property
    def max_number_of_shots(self) -> int:
        if type(self.number_of_shots) is int:
            return self.number_of_shots
        return int(max(self.number_of_shots))

    @cached_property
    def min_number_of_shots(self) -> int:
        if type(self.number_of_shots) is int:
            return self.number_of_shots
        return int(min(self.number_of_shots))

    @cached_property
    def number_of_shots_per_class(self) -> np.ndarray[int]:
        if type(self.number_of_shots) is int:
            return np.full(shape=self.number_of_ways, fill_value=self.number_of_shots)
        return self.number_of_shots


@dataclass
class Task:
    """
    Dataset name this task belongs to.
    """
    dataset_name: str
    """
    Support set used for training / finetuning of the model.
    """
    support_set: SetData
    """
    Query set used for prediction.
    """
    query_set: SetData

    # TODO: Probably should be inferred from support / query set
    """
    Number of ways.
    """
    number_of_ways: int
    """
    Original class names used in the task.
    """
    class_names: list[str]
