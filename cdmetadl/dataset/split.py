__all__ = ["random_meta_split", "random_class_split"]

from collections import defaultdict
import numpy as np

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


def rand_conf_split(support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], query_set: support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], lengths: list[float], num_ways: int, num_shots: int, seed: int = None) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: 
    if not np.isclose(sum(lengths), 1.0):
        raise ValueError("Sum of lengths must be approximately 1")

    if support_set[3].shape[1] < len(lengths): 
        raise ValueError("Not enough shots for split, should be at least", len(lengths))    

    print("shape support: ", support_set.shape)
    # reshape support and test set so that ways, shots can be accessed
    rearranged_support = [support_set[0].reshape(num_ways, num_shots, 3, 128, 128), support_set[1].reshape(num_ways, num_shots), support_set[2].reshape(num_ways, num_shots)]
    # matrix dim: 3 x num_ways, num_shots 
    rearranged_query = [query_set[0].reshape(num_ways, num_shots, 3, 128, 128), query_set[1].reshape(num_ways, num_shots)]
    print("shape rearranged support: ", rearranged_support.shape)
    print("rearranged_support", rearranged_support)
    nr_splits = len(lengths)

    # generate list of n_way*n_shot indices, shuffle them
    # n x way x 2xshot/2
    # split into 2 subarrays
    indices_per_way = [
            np.array_split(np.random.shuffle(np.arange(num_shots)), 2) for cls in num_ways
        ]

    print("indices_per_way", indices_per_way)
    # replace this with calculation from "nr_splits"
    cut_idxs = [np.floor(num_shots/2), num_shots]
    print("cut_idxs", cut_idxs)

    split_support = [[[[rearranged_support[i][j][idx] for idx in indices[s]] for j, indices in enumerate(indices_per_way)] for i in range(len(rearranged_support))] for s in range(nr_splits)]
    # matrix with dim: 2x3xnum_ways x num_shots
    print("split_support", split_support.shape)
    
    split_query = [[[[rearranged_query[i][j][idx] for idx in indices[s]] for j, indices in enumerate(indices_per_way)] for i in range(len(rearranged_query))] for s in range(nr_splits)]
    # matrix with dim: 2x2xnum_waysxnum_shots

    arranged_split_support = [el.reshape(num_ways, el.shape[2]) for el in split_support]
    arranged_split_test = [el.reshape(num_ways, el.shape[2]) for el in split_query]

    return arranged_split_support, arranged_split_test 
