__all__ = ["DataGenerator", "BatchGenerator", "SampleTaskGenerator", "TaskGenerator"]

import abc
from typing import Iterator
import torch
import torch.utils.data

import cdmetadl.config
import cdmetadl.samplers

from .meta_image_dataset import MetaImageDataset
from .task import Task


class DataGenerator(abc.ABC):

    def __init__(self, dataset: MetaImageDataset, config: cdmetadl.config.DatasetConfig):
        self.dataset = dataset
        self.config = config
        self.total_number_of_classes = self.dataset.total_number_of_classes

    @property
    @abc.abstractmethod
    def number_of_classes(self) -> int:
        pass


class BatchGenerator(DataGenerator):

    def __call__(self, num_batches: int) -> Iterator:
        generated_batches = 0
        while True:
            for batch in torch.utils.data.DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True):
                if generated_batches == num_batches:
                    return
                yield batch
                generated_batches += 1

    @property
    def number_of_classes(self) -> int:
        return self.dataset.total_number_of_classes


class SampleTaskGenerator(DataGenerator):

    def __call__(self, num_tasks: int) -> Iterator[Task]:
        yield from self.dataset.generate_tasks(
            num_tasks, self.config.n_ways, self.config.k_shots, self.config.query_size
        )

    @property
    def number_of_classes(self) -> int:
        if isinstance(self.config.n_ways, cdmetadl.samplers.ValueSampler):
            return self.config.n_ways.value
        raise ValueError(
            "Sampler with variable number of ways is used in task mode, thus number of classes cannot be determined"
        )


class TaskGenerator(DataGenerator):

    def __call__(self, num_tasks_per_dataset: int) -> Iterator[Task]:
        yield from self.dataset.generate_tasks_for_each_dataset(
            num_tasks_per_dataset, self.config.n_ways, self.config.k_shots, self.config.query_size
        )

    @property
    def number_of_classes(self) -> int:
        if isinstance(self.config.n_ways, cdmetadl.samplers.ValueSampler):
            return self.config.n_ways.value
        raise ValueError(
            "Sampler with variable number of ways is used in task mode, thus number of classes cannot be determined"
        )
