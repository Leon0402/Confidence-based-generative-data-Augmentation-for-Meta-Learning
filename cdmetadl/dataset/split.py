__all__ = ["random_meta_split"]

import numpy as np

from .meta_image_dataset import MetaImageDataset


def random_meta_split(dataset: MetaImageDataset, lengths: list[float]) -> list[MetaImageDataset]:
    if not np.isclose(sum(lengths), 1.0):
        raise ValueError("Sum of lengths must be approximately 1")

    number_of_datasets = len(dataset.datasets)
    shuffled_dataset_indices = np.random.permutation(number_of_datasets)

    split_indices = np.ceil(np.cumsum(lengths) * number_of_datasets).astype(int)

    return [
        MetaImageDataset([dataset.datasets[index]
                          for index in indices])
        for indices in np.split(shuffled_dataset_indices, split_indices[:-1])
    ]


# TODO(leon): Return MetaImageDataset rather than Subset
# def random_class_split(dataset: MetaImageDataset, lengths: list[float]) -> list[torch.utils.data.Subset]:
#     if not np.isclose(sum(lengths), 1.0):
#         raise ValueError("Sum of lengths must be approximately 1")

#     if len(dataset.datasets) != 1:
#         raise ValueError("Meta dataset should contain exactly one image dataset for class splitting")

#     image_dataset = dataset.datasets[0]

#     shuffled_classes = np.random.permutation(image_dataset.number_of_classes)
#     split_indices = np.cumsum(np.array(lengths) * image_dataset.number_of_classes).astype(int)

#     return [
#         torch.utils.data.Subset(dataset, np.concatenate([image_dataset.idx_per_label[i]
#                                                          for i in class_subset]))
#         for class_subset in np.split(shuffled_classes, split_indices[:-1])
#     ]
