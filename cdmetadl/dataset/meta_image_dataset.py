__all__ = ["MetaImageDataset"]

from typing import Iterator
import bisect
import itertools
import random

import torch.utils.data

import cdmetadl.samplers

from .image_dataset import ImageDataset
from .task import Task


class MetaImageDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for handling multiple image datasets from MetaAlbum in PyTorch. 
    
    Attributes:
        datasets (list[ImageDataset]): List of datasets this meta dataset holds.
        dataset_end_index (list[int]): Lift of end indices of the datasets (non inclusive).
    """

    def __init__(self, datasets: list[ImageDataset]):
        """
        TODO
        """
        self.datasets = datasets
        self.number_of_datasets = len(self.datasets)
        self.dataset_end_index = list(itertools.accumulate(len(dataset) for dataset in self.datasets))
        self.total_number_of_classes = sum(dataset.number_of_classes for dataset in self.datasets)

    def __len__(self) -> int:
        """
        Returns the number of images in the dataset.

        Returns:
            int: Total number of images.
        """
        return self.dataset_end_index[-1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the image and its label at a given index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the image and its label, both as tensors.
        """
        dataset_idx, local_idx = self._calculate_2d_index_from_1d(idx)
        return self.datasets[dataset_idx][local_idx]

    def _calculate_2d_index_from_1d(self, idx: int) -> tuple[int, int]:
        """
        Given an one dimensional index it calculates a 2D index indicating the dataset and the index within that dataset.

        Args:
            idx (int): 1D index which should be converted.

        Returns:
            tuple[int, int]: A tuple containing the index of the dataset and the index within the dataset.
        """
        dataset_idx = bisect.bisect_left(self.dataset_end_index, idx + 1)
        if dataset_idx == 0:
            return 0, idx
        return dataset_idx, idx - self.dataset_end_index[dataset_idx - 1]

    def generate_tasks(
        self, num_tasks: int, n_ways: cdmetadl.samplers.Sampler, k_shots: cdmetadl.samplers.Sampler, query_size: int
    ) -> Iterator[Task]:
        for _ in range(num_tasks):
            dataset = random.choice(self.datasets)
            yield dataset.generate_task(n_ways, k_shots, query_size)

    def generate_tasks_for_each_dataset(
        self, num_tasks_per_dataset: int, n_ways: cdmetadl.samplers.Sampler, k_shots: cdmetadl.samplers.Sampler,
        query_size: int
    ) -> Iterator[Task]:
        for dataset in self.datasets:
            for _ in range(num_tasks_per_dataset):
                yield dataset.generate_task(n_ways, k_shots, query_size)
