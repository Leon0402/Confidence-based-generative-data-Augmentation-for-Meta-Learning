__all__ = ["split_by_names", "random_meta_split", "random_class_split", "set_split"]

import numpy as np
import torch
import random

from .meta_image_dataset import MetaImageDataset
from .image_dataset import ImageDataset
from .task import SetData


def split_by_names(meta_dataset: MetaImageDataset, names: list[list[str]]) -> list[MetaImageDataset]:
    return [
        MetaImageDataset([dataset
                          for dataset in meta_dataset.datasets
                          if dataset.name in dataset_names])
        for dataset_names in names
    ]


def random_meta_split(meta_dataset: MetaImageDataset, lengths: list[float], seed: int = None) -> list[MetaImageDataset]:
    if not np.isclose(sum(lengths), 1.0):
        raise ValueError("Sum of lengths must be approximately 1")

    number_of_datasets = len(meta_dataset.datasets)
    shuffled_dataset_indices = np.random.default_rng(seed=seed).permutation(number_of_datasets)

    split_indices = np.ceil(np.cumsum(lengths) * number_of_datasets).astype(int)

    return [
        MetaImageDataset([meta_dataset.datasets[index]
                          for index in indices])
        for indices in np.split(shuffled_dataset_indices, split_indices[:-1])
    ]


def random_class_split(meta_dataset: MetaImageDataset, lengths: list[float],
                       seed: int = None) -> list[MetaImageDataset]:
    if not np.isclose(sum(lengths), 1.0):
        raise ValueError("Sum of lengths must be approximately 1")

    filtered_datasets_by_split = {idx: [] for idx in range(len(lengths))}

    for dataset in meta_dataset.datasets:
        shuffled_classes = np.random.default_rng(seed=seed).permutation(list(dataset.label_names))
        split_indices = np.cumsum(np.array(lengths) * dataset.number_of_classes).astype(int)

        for split_idx, class_subset in enumerate(np.split(shuffled_classes, split_indices[:-1])):
            filtered_datasets_by_split[split_idx].append(
                ImageDataset(dataset.name, dataset.dataset_info, dataset.img_size, included_classes=set(class_subset))
            )

    return [MetaImageDataset(datasets) for datasets in filtered_datasets_by_split.values()]


def set_split(data_set: SetData, number_of_splits: int) -> SetData:
    if number_of_splits > data_set.number_of_shots:
        raise ValueError(
            f"Number of splits {number_of_splits} cannot be greater than number of shots {data_set.number_of_shots}"
        )

    images = data_set.images_by_class.swapaxes(0, 1)
    labels = data_set.labels_by_class.swapaxes(0, 1)

    base_size, remainder = divmod(data_set.number_of_shots, number_of_splits)
    split_sizes = [base_size] * number_of_splits
    for i in range(remainder):
        split_sizes[i] += 1

    return [
        SetData(
            images=images_split.swapaxes(0, 1).flatten(end_dim=1),
            labels=labels_split.swapaxes(0, 1).flatten(end_dim=1),
            number_of_ways=data_set.number_of_ways,
            numer_of_shots=split_sizes,
        ) for images_split, labels_split in zip(torch.split(images, split_sizes), torch.split(labels, split_sizes))
    ]
