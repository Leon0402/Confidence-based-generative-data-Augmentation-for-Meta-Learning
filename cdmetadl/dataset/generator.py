import torch

import cdmetadl.dataset


def create_batch_generator(dataset: cdmetadl.dataset.MetaImageDataset):

    def batch_generator(num_batches: int):
        return iter(cdmetadl.helpers.general_helpers.cycle(num_batches, torch.utils.data.DataLoader(dataset)))

    return batch_generator


def create_task_generator(dataset: cdmetadl.dataset.MetaImageDataset, sample_dataset: bool = False):

    def sample_task_generator(num_tasks: int):
        yield from dataset.generate_tasks()

    def task_generator(num_tasks: int):
        yield from dataset.generate_tasks_for_each_dataset()

    return sample_task_generator if sample_dataset else task_generator
