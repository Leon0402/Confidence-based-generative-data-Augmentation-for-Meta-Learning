from typing import Iterator
import bisect
import itertools
import random

import torch.utils.data

import cdmetadl.dataset


class MetaImageDataset(torch.utils.data.Dataset):

    def __init__(self, datasets_info):
        self.datasets = [cdmetadl.dataset.ImageDataset(dataset_info) for dataset_info in datasets_info]
        self.dataset_end_index = list(itertools.accumulate(len(dataset) for dataset in self.datasets))

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
        dataset_idx = bisect.bisect_left(self.dataset_end_index, idx)
        local_idx = idx - self.dataset_end_index.get(dataset_idx - 1, 0)
        return self.datasets[dataset_idx][local_idx]

    def generate_tasks(self, num_tasks: int, n_way: int, k_shot: int,
                       query_size: int) -> Iterator[cdmetadl.dataset.Task]:
        for _ in range(num_tasks):
            dataset = random.choice(self.datasets)
            yield dataset.generate_task()

    def generate_tasks_for_each_dataset(self, num_tasks_per_dataset: int, n_way: int, k_shot: int,
                                        query_size: int) -> Iterator[cdmetadl.dataset.Task]:
        for dataset in self.datasets:
            for _ in range(num_tasks_per_dataset):
                yield dataset.generate_task()

def create_batch_generator(dataset: cdmetadl.dataset.MetaImageDataset)
    def batch_generator(num_batches: int):
        return iter(cdmetadl.helpers.general_helpers.cycle(num_batches, torch.utils.data.DataLoader(dataset)))
    
    return batch_generator

def create_task_generator(dataset: cdmetadl.dataset.MetaImageDataset):

    def task_generator(num_tasks: int):
        yield from dataset.generate_tasks()
    
    return task_generator