from typing import Self
import torch



class Augmentation():
    def __init__(self, dataset: MetaImageDataset, confidence_threshold: int, conf_scores: list[float]):
        self.dataset = dataset
        self.confidence_threshold = confidence_threshold
        self.conf_scores = conf_scores
       

class StandardAug(Augmentation):

    def __init__(self, dataset: MetaImageDataset, confidence_threshold: int, conf_scores: list[float]):
        super().__init__(dataset, confidence_threshold, conf_scores)




class PseudoAug(Augmentation):
# get during meta testing, support 

    def __init__(self, dataset: MetaImageDataset, confidence_threshold: int, conf_scores: list[float]):
        super().__init__(dataset, confidence_threshold, conf_scores)
    
    def get_augmented_DS(): 
        for score in self.conf_scores: 
            if score < self.confidence_threshold:        


class GenerativeAug(Augmentation):
    def __init__(self, dataset: MetaImageDataset, confidence_threshold: int, conf_scores: list[float]):
        super().__init__(dataset, confidence_threshold, conf_scores)
       