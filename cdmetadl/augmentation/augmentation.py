import torch

import cdmetadl.dataset
import cdmetadl.helpers.general_helpers

# TODO: make more modular class structure and useful methods
class Augmentation():
    
    def __init__(self, support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], conf_scores: list[float], threshold: float, scale: int):
        self.support_set = support_set
        self.conf_support_set = conf_support_set
        self.conf_scores = conf_scores
        self.threshold = threshold
        self.scale = scale
       


class PseudoAug(Augmentation):

    """
    Class for pseudo augmentation. Goes through all ways/classes, checks confidence score for particular class, calculates number of extra samples to sample for particular class as an inverse of its 
    confidence score * scale rounded down. It will randomly pick samples from the backup set and concatenate them with the original support_set. It also augments the label and original label tensors accordingly
    to account for variable shots. 

    Args:
        support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Support set used for pretraining the model on.
        backup_support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Query set for prediction after pretraining and estimation of confidence scores through softmax. 
        conf_scores: (MyLeaner) Pretrained model instatiated for meta-testing. 
        threshold: (float) Confidence value below which we want to augment the class. 
        scale: (int) Indicates how much in "fold" of the orginal set we want to add to the augmented one. 
        num_ways: (int) Number of ways in original task. 
        num_shots: (int) Number of shots in original task. 

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor: Augmented 
    """

    def __init__(self, support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], backup_support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], conf_scores: list[float], threshold: float, scale: int, num_shots: int, num_ways: int):
        super().__init__(support_set, conf_scores, threshold, scale)
        self.backup_support_set = backup_support_set

        rearranged_conf_support = [backup_support_set[0].reshape(num_ways, num_shots, 3, 128, 128), backup_support_set[1].reshape(num_ways, num_shots), backup_support_set[2].reshape(num_ways, num_shots)]

    
    def getDatasetAugmented(): 
        shots = list()
        samples_idxs = list()

        # go through all classes and check scores vs threshold per class
        for idx, score in enumerate(self.conf_scores): 

            if score < self.threshold:   
                # calculate amount of samples to be added and augmented with      
                nr_samples = 1/score * nr_shots * scale
                # add number of total shots for this class in augmented dataset
                shots[idx] = nr_samples + num_shot
                # get nr_samples random indexes for sampling from backup_support_set
                sample_idxs.append(np.random.choice(0, nr_samples))
                
        support_images = torch.stack([
                rearranged_conf_support[idx][i] for i in way_idxs for j, way_idxs in enumerate(sample_idxs)
        ])

        # concat support and augmented dataset, adjust labels and original labels to account for variable shots
        augmented_support_set = (support_set[0].cat(support_images), torch.tensor(np.arange(n_way).repeat(shots)), torch.tensor(selected_classes.repeat(shots)))
        # TODO: check dims of this return
        return augmented_support_set, shots




class StandardAug(Augmentation):

    def __init__(self, task: Task, confidence_threshold: float, aug_factor: int, conf_scores: list[float]):
        super().__init__(task, confidence_threshold, aug_factor, conf_scores)



class GenerativeAug(Augmentation):
    def __init__(self, task: Task, confidence_threshold: float, aug_factor: int, conf_scores: list[float]):
        super().__init__(task, confidence_threshold, aug_factor, conf_scores)
       

    