__all__ = ["Augmentation", "PseudoAug", "StandardAug", "GenerativeAug"]

import torch
import numpy as np

import cdmetadl.dataset
import cdmetadl.helpers.general_helpers

import abc


# TODO: make more modular class structure and useful methods
class Augmentation(metaclass=abc.ABCMeta):
    def __init__(self, threshold: float, scale: int):
        self.threshold = threshold
        self.scale = scale

       
    @abc.abstractmethod
    def augment(self, support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], conf_scores: list[float], backup_support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], num_ways: int): 
        pass 

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
        num_ways: (int) Number of ways in backup_support set. 
        num_shots: (int) Number of shots in backup_support set. 

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor: Augmented 
    """

    def augment(self, support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], conf_scores: list, backup_support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], num_ways: int): 
        shots = list()
        sample_idxs = list()
        num_shots_support = int(support_set[1].shape[0] / num_ways)
        num_shots_support_backup = int(support_set[1].shape[0] / num_ways) #backup_support_set[1].shape[0] / num_ways
        print("-------------------------------------------------------------------------------------------")
        print("shots: ", num_shots_support, num_shots_support_backup)
       # print("backup images", backup_support_set[0][0])
        #print("images", support_set[0][0])
        # fix dims
        #rearranged_backup_images = backup_support_set[0].reshape(num_shots_support, num_ways, 3, 128, 128)
        # should be np.transpose(reshape(num_ways, num_shots))
        rearranged_backup_images = support_set[0].reshape(num_shots_support, num_ways, 3, 128, 128)
        rearranged_images = support_set[0].reshape(num_shots_support, num_ways, 3, 128, 128)

        rearranged_images_ways = support_set[0].reshape(num_ways, num_shots_support, 3, 128, 128)
        #print("backup_im dim", rearranged_backup_images[0][0], rearranged_backup_images.shape)
        # must be the same
        print("original image", support_set[0][0])
        print("rearranged_images_ways", rearranged_images_ways.shape, rearranged_images_ways[0][0].shape, rearranged_images_ways[0][0])
        # by shots indexed
        print("rearranged_images", rearranged_images.shape, rearranged_images[0][0].shape, rearranged_images[0][0])
        # get original labels 
        original_labels = np.array([support_set[2][i*num_shots_support].item() for i in range(num_ways)])

        # go through all classes and check scores vs threshold per class
        for clls, score in enumerate(conf_scores): 
            print("score, class", score, clls)
            if score < self.threshold:   
                # calculate amount of samples to be added and augmented with      
                nr_samples = 1/score * num_shots_support * self.scale

                # if number of samples ought to be greater than the available samples per shot from backup_set, set it to the max
                if nr_samples > num_shots_support_backup: 
                    nr_samples = num_shots_support_backup
                # add number of total shots for this class in augmented dataset
                shots.append(nr_samples + num_shots_support)
                # get nr_samples random indexes for sampling from backup_support_set
                sample_idxs.append(np.random.choice(np.arange(0, nr_samples), nr_samples))
            else: 
                shots.append(num_shots_support)    
                sample_idxs.append(())

        print("shots list", shots)  
        print("samples idxs", sample_idxs, len(sample_idxs))     

        
        # build tensor together as whole not with concat
        # way x shots of sampled backup images 
        # gives matrix num_shots x num_ways including, for every way, the images from the original tensor, appended with the sampled images from the backup support set
        #images = [np.concatenate((rearranged_images_ways[j], [rearranged_backup_images[i][j] for i in way_idxs]), axis=0) for j, way_idxs in enumerate(sample_idxs)]
        images_way = list()
        for j, way_idxs in enumerate(sample_idxs): 
            backup_images_way = np.array([rearranged_backup_images[i][j] for i in way_idxs])
            original_images_way = rearranged_images_ways[j]
            print("way: ", j)
            print("way_idx", way_idxs)
            print("backup_images_way dim", backup_images_way.shape)
            print("original_images_way dim", original_images_way.shape)
            if not len(way_idxs) == 0:
                images_way.append(np.concatenate((original_images_way, backup_images_way), axis=0))
            else:
                images_way.append(original_images_way)  
                    
    
        # concat support and augmented dataset, adjust labels and original labels to account for variable shots
        # TODO: dont concat but shuffle, reshape
       
        #print(len(images_way), len(images_way[0]), len(images_way[1]), len(images_way[2]), len(images_way[3]))
        images = np.array([images_way[j][i] for j in range(len(images_way)) for i in range(len(images_way[j])) ])
        
        print("images.shape", images.shape, images[0].shape, images[0])
        augmented_support_set = (torch.tensor(images), torch.tensor(np.arange(num_ways).repeat(shots)), torch.tensor(original_labels.repeat(shots)))
       # print("augmented_support_set", augmented_support_set[1], augmented_support_set[2], augmented_support_set[0].shape, augmented_support_set[0])
  
        return augmented_support_set, shots




class StandardAug(Augmentation):
    def augment(self, support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], conf_scores: list[float], backup_support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None): 
        pass
    



class GenerativeAug(Augmentation):
    def augment(self, support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], conf_scores: list[float], backup_support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None): 
        pass

    