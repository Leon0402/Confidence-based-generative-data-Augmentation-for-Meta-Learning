import numpy as np
import torch.utils.data

import cdmetadl.helpers.general_helpers
import cdmetadl.dataset


def random_meta_split(
    dataset: cdmetadl.dataset.MetaImageDataset,
    lengths: list[float],
    generator: torch.Generator = torch.default_generator
) -> list[torch.utils.data.Subset]:
    if not np.isclose(sum(lengths), 1.0):
        raise ValueError("Sum of lengths must be approximately 1")

    number_of_datasets = len(dataset.datasets)
    shuffled_dataset_indices = np.random.default_rng(generator).permutation(number_of_datasets)

    proportions = np.cumsum(lengths) * number_of_datasets
    split_indices = np.searchsorted(proportions, np.arange(1, number_of_datasets)).tolist()

    return [
        torch.utils.data.Subset(dataset, indices.tolist())
        for indices in np.split(shuffled_dataset_indices, split_indices)
    ]


def random_class_split(dataset: cdmetadl.dataset.MetaImageDataset,
                       lengths: list[float],
                       generator=None) -> list[torch.utils.data.Subset]:
    if not np.isclose(sum(lengths), 1.0):
        raise ValueError("Sum of lengths must be approximately 1")

    if len(dataset.datasets) != 1:
        raise ValueError("Meta dataset should contain exactly one image dataset for class splitting")

    number_of_classes = len(dataset.datasets[0].idx_per_label)
    shuffled_classes = np.random.default_rng(generator).permutation(number_of_classes)

    split_indices = np.cumsum(np.array(lengths) * number_of_classes).astype(int)
    split_indices[-1] = number_of_classes  # Ensure the last index is the total number of classes

    return [
        torch.utils.data.Subset(dataset, np.concatenate([dataset.datasets[0].idx_per_label[i]
                                                         for i in class_subset]))
        for class_subset in np.split(shuffled_classes, split_indices[:-1])
    ]
