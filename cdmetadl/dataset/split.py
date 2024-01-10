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

   
    # reshape input array so that number of shots is first index and it can be split based on that
    # output dim num_shots x num_ways x 3 x 128 x 128 for images
    rearranged_support = support_set[0].reshape(num_shots, num_ways, 3, 128, 128)

    query_size = int(query_set[0].shape[0] / num_ways)
    rearranged_query = query_set[0].reshape(query_size, num_ways, 3, 128, 128)
    original_labels = np.array([support_set[2][i*num_shots].item() for i in range(num_ways)])
    print("shape support set", support_set[0].shape)
    print("shape query set", query_set[0].shape)

    print("reshapedshape support set", rearranged_support.shape)
    print("shape query set", rearranged_query.shape)
    
    nr_splits_query = 2

    # for every class randomize indices and split them based on nr_splits
    # nr_splits for support can be 2 e.g in case no backup set is needed for augmentation
    indices_per_way_query = [
        np.array_split(random.sample(list(np.arange(query_size)), query_size), nr_splits_query) for cls in range(num_ways)
        ]

    indices_per_way_support = [
        np.array_split(random.sample(list(np.arange(num_shots)), num_shots), nr_splits_support) for cls in range(num_ways)
        ]    

    print("indices per way support", indices_per_way_support)
    print("indices per way query", indices_per_way_query)
    # gives back the split image tensors in proper dimensions nr_splits x n_ways*n_shots
    # split_nr x way*shot 
  
    split_support_images = [[rearranged_support[i][j] for j, indices in enumerate(indices_per_way_support) for i in indices[split]] for split in range(nr_splits_support)]
    split_query_images = [[rearranged_query[i][j] for j, indices in enumerate(indices_per_way_query) for i in indices[split]] for split in range(nr_splits_query)]

  
    print("SPLIT IMAGES", len(split_support_images[0]), len(split_support_images[0][0]), len(split_support_images[0][0][0]))
    print("split_uery_images", split_support_images[0][0])
   
    support_shots_nr = np.array([int(len(split))/num_ways for split in split_support_images])
    query_shots_nr = np.array([int(len(split))/num_ways for split in split_query_images])

    print("DEBUG support_shots_nr", support_shots_nr, support_shots_nr.shape)
    print("DEBUG query_shots_nr", query_shots_nr, query_shots_nr.shape)

    # dim nr_split x 3 x ((n_ways*n_split_shots, 3, 128, 128), (n_ways*n_split_shots), (n_ways*n_split_shots))
    # TODO: classes wrong for 3rd tensor, same with query
    split_support = [(torch.tensor(np.array(split_support_images[i])), torch.tensor(np.arange(num_ways).repeat(support_shots_nr[i])),
                torch.tensor(original_labels.repeat(support_shots_nr[i]))) for i in range(nr_splits_support)]


    split_query = [(torch.tensor(np.array(split_query_images[i])), torch.tensor(np.arange(num_ways).repeat(query_shots_nr[i])),
                torch.tensor(original_labels.repeat(query_shots_nr[i]))) for i in range(nr_splits_query)]

    return split_support[0], split_support[1], split_support[2], split_query[0], split_query[1]
    

