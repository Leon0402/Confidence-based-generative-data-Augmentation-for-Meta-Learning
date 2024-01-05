__all__ = ["Task"]

from dataclasses import dataclass

import torch
import numpy as np


@dataclass
class Task:
    """ Class to define few-shot learning tasks.

    Args:
            num_ways (int): Number of ways (classes) in the current task. 
            num_shots (int): Number of shots (images per class) for the support 
                set.
            variable_shots: (boolean): Boolean indicating whether shots are 
                variable in this task, needed for augmented tasks where this is the 
                case. Defaults to False. 
            num_shots_variable: (list[int]): In case of variable shots, this contains
                the number of shots per class. 
                TODO: add default here.
            support_set (SET_DATA): Support set for the current task. The 
                format of the set is (torch.Tensor, torch.Tensor, torch.Tensor)
                where the first tensor corresponds to the images with a shape 
                of [num_ways*num_shots x 3 x 128 x 128]. The second tensor 
                corresponds to the labels with a shape of [num_ways*num_shots].
                The last tensor corresponds to the original labels with a shape 
                of [num_ways*num_shots].
            query_set (SET_DATA): Query set for the current task. The format
                of the set is (torch.Tensor, torch.Tensor, torch.Tensor), where 
                the first tensor corresponds to the images with a shape of 
                [num_ways*query_size x 3 x 128 x 128]. The second tensor 
                corresponds to the labels with a shape of [num_ways*query_size]
                and the last tensor corresponds to the original labels with a 
                shape of [num_ways*num_shots]. The query_size can vary 
                depending on the configuration of the data loader.
            original_class_idx (np.ndarray): Array with the original class 
                indexes used in the current task, its shape is [num_ways, ].
            dataset (str, optional): Name of the dataset used to create the 
                current task. Defaults to None.
                
    """
    num_ways: int
    num_shots: int
    variable_shots = False
    num_shots_variable: list[int]
    support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    query_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    original_class_idx: np.ndarray
    dataset: str = None