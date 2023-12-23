__all__ = ["DataGenerator", "BatchGenerator", "TaskGenerator"]

import torch
import torch.utils.data

from .meta_image_dataset import MetaImageDataset


class DataGenerator():

    def __init__(self, dataset: MetaImageDataset, number_of_classes: int):
        self.dataset = dataset
        self.total_number_of_classes = self.dataset.total_number_of_classes
        self.number_of_classes = number_of_classes


class BatchGenerator(DataGenerator):

    def __init__(self, dataset: MetaImageDataset, config: dict):
        super().__init__(dataset, dataset.total_number_of_classes)
        self.config = config

    def __call__(self, num_batches: int):
        generated_batches = 0
        while True:
            for batch in torch.utils.data.DataLoader(self.dataset, batch_size=self.config["batch_size"], shuffle=True):
                if generated_batches == num_batches:
                    return
                yield batch
                generated_batches += 1

class TaskGenerator(DataGenerator):

    def __init__(self, dataset: MetaImageDataset, config: dict, sample_dataset: bool = False):
        super().__init__(dataset, config["N"] or None)
        self.sample_dataset = sample_dataset

        self.min_N = config["N"] or config["min_N"]
        self.max_N = config["N"] or config["max_N"]
        self.min_k = config["k"] or config["min_k"]
        self.max_k = config["k"] or config["max_k"]
        self.query_size = config["query_images_per_class"]

        self.classes = self.min_N if self.min_N == self.max_N else None

    def __call__(self, num_tasks: int):
        if self.sample_dataset:
            yield from self.dataset.generate_tasks(
                num_tasks, self.min_N, self.max_N, self.min_k, self.max_k, self.query_size
            )
        else:
            yield from self.dataset.generate_tasks_for_each_dataset(
                num_tasks, self.min_N, self.max_N, self.min_k, self.max_k, self.query_size
            )

