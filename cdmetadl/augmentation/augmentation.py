import torch

import cdmetadl.dataset
import cdmetadl.helpers.general_helpers

class Augmentation():
    
    def __init__(self, support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], conf_support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], conf_scores: list[float], threshold: float, scale: int):
        self.support_set = support_set
        self.conf_support_set = conf_support_set
        self.conf_scores = conf_scores
        self.threshold = threshold
        self.scale = scale
       


class PseudoAug(Augmentation):
# conf_scores: get confidence scores calculated from reference set (which is a "horizontal" split of image_dataset of test set) averaged over all tasks as well as particular task 
# generated by testgenerator
# aug_factor: if confidence is below a threshold, how many images to we want to add relative to the ones we have e.g factor of 2 means double the images for classes, could be variable and based on how low confidence actually is. total number of samples in task rounded if uneven
# task: test task from testgenerator of N*K*image_dim dimension
# confidence_threshold: float indicating below which confidence value to do augmentation
# returns modified augmentated_task(which inherits from task) with added images sampled from image_dataset the task is from (maybe pass this information)
# could contain dubplicates in one test augmented_task due to sampling randomness
    
    def getDatasetAugmented(): 
        for idx, score in enumerate(self.conf_scores): 
            if score < self.threshold:        
            # randomly sample: scale * 1/score * nr_shots from this class/way from conf_support_set
            # add to support_set and return augmented dataset  



class StandardAug(Augmentation):

    def __init__(self, task: Task, confidence_threshold: float, aug_factor: int, conf_scores: list[float]):
        super().__init__(task, confidence_threshold, aug_factor, conf_scores)



class GenerativeAug(Augmentation):
    def __init__(self, task: Task, confidence_threshold: float, aug_factor: int, conf_scores: list[float]):
        super().__init__(task, confidence_threshold, aug_factor, conf_scores)
       







def augmentTask(task: Task, extension_set: Image_Dataset, confidence_threshold: float, aug_factor: int, conf_scores: list[float]): 
    """ 
    Called in meta-testing loop, receives informaiton form the confidence estimation, current task to augment and by how much, dataset to augment with (prepared ahead)
    and constructs dict extracting class information from confidence scores and threshold. 
    Calls augment_task on dataset. 
    Returns augmented task with variable number of shots. 

    Args:
            task (Task): Task that will be augmented. 
            extension_set: (ImageDataset) Synthetic dataset (in case of GAN of conv. DA or split of test set in case of pseudo DA) task gets augmented with. 
            confidence_threshold: (float) Threshold below which classes should be augmented. 
            aug_factor: (int) Value indicating by how many samples task should be augmented, relative to the samples it already contains per shot. 
            conf_scores: (list[float]): Confidence scores for all classes in task as calculated in eval.py in confidence estimation step. 
    Returns: 
            task(Task): calls augmentTask in ImageData for dataset and returns augmented task. 
                
    """
    