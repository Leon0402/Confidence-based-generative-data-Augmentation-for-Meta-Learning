__all__ = ["random_meta_split", "random_class_split", "rand_conf_split"]

from collections import defaultdict
import numpy as np
import torch
import random

from .meta_image_dataset import MetaImageDataset
from .image_dataset import ImageDataset


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






# TODO: bug fix here
def rand_conf_split(support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], query_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], num_ways: int, num_shots: int, nr_splits_support: int = 3, seed: int = None): 

    # reshape input array so that 
    rearranged_support = [np.transpose(support_set[0]).reshape(num_shots, num_ways, 3, 128, 128), np.transpose(support_set[1].reshape(num_ways, num_shots)), np.transpose(support_set[2].reshape(num_ways, num_shots))]

    query_size = int(query_set[0].shape[0] / num_ways)

    rearranged_query = [np.transpose(query_set[0]).reshape(query_size, num_ways, 3, 128, 128), np.transpose(query_set[1]).reshape(query_size, num_ways)]
    
    nr_splits_query = 2


    indices_per_way_query = [
        np.array_split(random.sample(list(np.arange(num_shots)), num_shots), nr_splits_query) for cls in range(num_ways)
        ]

    indices_per_way_support = [
        np.array_split(random.sample(list(np.arange(num_shots)), num_shots), nr_splits_support) for cls in range(num_ways)
        ]    

    print("indices per way, shuffled and split: ", indices_per_way_support[0])


    #split_support = [[[[rearranged_support[i][j][idx] for idx in indices[s]] for j, indices in enumerate(indices_per_way_support)] for i in range(len(rearranged_support))] for s in range(nr_splits_support)]
    # TODO: rewrite this in easier to understand way
    split_query = [[[[rearranged_query[i][j][idx] for idx in indices[s]] for j, indices in enumerate(indices_per_way_query)] for i in range(len(rearranged_query))] for s in range(nr_splits_query)]


    split_support = [[support_set[0].reshape(len(support_set[0][0])* num_ways, 3, 128, 128), support_set[1].reshape(num_ways*len(support_set[0][0])), support_set[2].reshape(len(support_set[0][0])*num_ways), num_ways, len(support_set[0][0])] for support_set in split_support]
    split_query = [[query_set[0].reshape(len(query_set[0][0]), num_ways, 3, 128, 128), query_set[1].reshape(len(query_set[0][0])*num_ways), num_ways, len(query_set[0][0])] for query_set in split_query]

    return split_support, split_query
    

