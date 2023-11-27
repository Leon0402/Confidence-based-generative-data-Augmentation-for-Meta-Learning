__all__ = ["create_batch_generator", "create_task_generator"]

from typing import Any, Iterator

import torch

import cdmetadl.helpers.general_helpers

from .meta_image_dataset import MetaImageDataset


def cycle(steps: int, iterable: Any) -> Iterator[Any]:
    """ Creates a cycle of the specified number of steps using the specified 
    iterable.

    Args:
        steps (int): Steps of the cycle.
        iterable (Any): Any iterable. In the ingestion program it is used when
            batch data format is selected for training.

    Yields:
        Iterator[Any]: The output of the iterable.
    """
    c_steps = -1
    while True:
        for x in iterable:
            c_steps += 1
            if c_steps == steps:
                return
            yield x


def create_batch_generator(dataset: MetaImageDataset):

    def batch_generator(num_batches: int):
        return iter(cycle(num_batches, torch.utils.data.DataLoader(dataset)))

    return batch_generator


def create_task_generator(dataset: MetaImageDataset, config: dict, sample_dataset: bool = False):
    min_N = config["N"] or config["min_N"]
    max_N = config["N"] or config["max_N"]
    min_k = config["k"] or config["min_k"]
    max_k = config["k"] or config["max_k"]
    query_size = config["query_images_per_class"]

    def sample_task_generator(num_tasks: int):
        yield from dataset.generate_tasks(num_tasks, min_N, max_N, min_k, max_k, query_size)

    def task_generator(num_tasks: int):
        yield from dataset.generate_tasks_for_each_dataset(num_tasks, min_N, max_N, min_k, max_k, query_size)

    return sample_task_generator if sample_dataset else task_generator
