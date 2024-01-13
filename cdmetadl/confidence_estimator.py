__all__ = ["ref_set_confidence_scores"]

import torch
import pandas as pd
import numpy as np

import cdmetadl.dataset
import cdmetadl.helpers.general_helpers
import cdmetadl.helpers.scoring_helpers
import cdmetadl.dataset.split

# TODO: fix this import
import sys
sys.path.append("..")
from baselines.finetuning.api import MetaLearner, Learner, Predictor
from baselines.maml.api import MetaLearner, Learner, Predictor

def ref_set_confidence_scores(conf_support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], conf_query_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], learner: Learner, num_ways: int) -> dict: 
    """
    Calculates confidence score for every class in task by pretraining/finetuning on validation/conf_support_set and prediction on validation query_set. Returns the mean of confidence scores per 
    class over every where a particular class was the correct predction. If class C is correctly classified, the CS will be the output of the softmax, if it is wrongly classified it will be 0. 

    Args:
        conf_support_set: (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Support set used for pretraining the model on.
        conf_query_set: (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Query set for prediction after pretraining and estimation of confidence scores through softmax. 
        learner: (MyLeaner) pretrained model instatiated for meta-testing. 

    Returns:
        list: list of confidence scores for all ways.
    """
    conf_scores = dict()
    num_shots = int(conf_support_set[1].shape[0] / num_ways)
    predictor = learner.fit((conf_support_set[0], conf_support_set[1], conf_support_set[2], num_ways, num_shots))
   
    predictions = predictor.predict(conf_query_set[0])
    ground_truth = conf_query_set[1].numpy()
    #print("ground truth", ground_truth)

    # go through predictions, compare with ground truth and update confidence score for particular class accordingly
    for idx, prediction in enumerate(predictions): 
        label = ground_truth[idx]
        #print("prediction for true class", label, prediction)

        if np.argmax(prediction) == label: 
            conf_sc = np.max(prediction)
        else: 
            conf_sc = 0.0    
        if label in conf_scores: 
            conf_scores[label] = np.mean([conf_scores[label], conf_sc])
        else: 
            conf_scores[label] = conf_sc
    return list(conf_scores.values())