import torch
import pandas as pd

import cdmetadl.dataset
import cdmetadl.helpers.general_helpers
import cdmetadl.helpers.scoring_helpers
import cdmetadl.dataset.split

# TODO: makes this more modular for other methods

def ref_set_confidence_scores(conf_support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], conf_query_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], learner: MyLearner) -> dict: 
    """
    Calculates confidence score for every class in task by pretraining/finetuning on validation/conf_support_set and prediction on validation query_set. Returns the mean of confidence scores per 
    class over every where a particular class was the correct predction. If class C is correctly classified, the CS will be the output of the softmax, if it is wrongly classified it will be 0. 

    Args:
        conf_support_set: (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Support set used for pretraining the model on.
        conf_query_set: (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Query set for prediction after pretraining and estimation of confidence scores through softmax. 
        learner: (MyLeaner) pretrained model instatiated for meta-testing. 

    Returns:
        dict: dictionary with key being class and value the confidence estimate for that class.
    """
    conf_scores = dict()
    learner.fit(support_set)
    # TODO: check for exact output of predictor.predict
    predictions = predictor.predict(query_set[0])
    # need softmax and label of predicted item
    print("predictions: ", predictions)

    # check if max(softmax) is of class same as ground truth -> add softmax to dict
    # otherwise add 0 to dict, keeping running average
    #for prediciton in predictions: 
    #    if 


    return conf_scores