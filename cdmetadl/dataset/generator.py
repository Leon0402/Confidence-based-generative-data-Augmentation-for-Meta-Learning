__all__ = ["DataGenerator", "BatchGenerator", "SampleTaskGenerator", "TaskGenerator"]

import torch
import torch.utils.data

import cdmetadl.config

from .meta_image_dataset import MetaImageDataset


class DataGenerator():

    def __init__(self, dataset: MetaImageDataset, config: cdmetadl.config.DatasetConfig):
        self.dataset = dataset
        self.config = config
        self.total_number_of_classes = self.dataset.total_number_of_classes

        match config.train_mode:
            case cdmetadl.config.DataFormat.BATCH:
                self.number_of_classes = dataset.total_number_of_classes
            case cdmetadl.config.DataFormat.TASK:
                # TODO(leon): Task Mode with Value Sampler => N
                self.number_of_classes = None


class BatchGenerator(DataGenerator):

    def __call__(self, num_batches: int):
        generated_batches = 0
        while True:
            for batch in torch.utils.data.DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True):
                if generated_batches == num_batches:
                    return
                yield batch
                generated_batches += 1


class SampleTaskGenerator(DataGenerator):

    def __call__(self, num_tasks: int):
        yield from self.dataset.generate_tasks(
            num_tasks, self.config.n_ways, self.config.n_ways, self.config.query_size
        )


class TaskGenerator(DataGenerator):

    def __call__(self, num_tasks_per_dataset: int):
        yield from self.dataset.generate_tasks_for_each_dataset(
            num_tasks_per_dataset, self.config.n_ways, self.config.n_ways, self.config.query_size
        )
