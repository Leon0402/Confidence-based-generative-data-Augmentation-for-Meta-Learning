import torch
import pandas as pd

import cdmetadl.dataset
import cdmetadl.helpers.general_helpers
import cdmetadl.helpers.scoring_helpers
import cdmetadl.dataset.split

# TODO: fix this import
import sys
sys.path.append("..")
from baselines.finetuning.api import MetaLearner, Learner, Predictor

# TODO: makes this more modular for other methods

def ref_set_confidence_scores(conf_support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], conf_query_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], learner: Learner) -> dict: 
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
   
    predictions = predictor.predict(query_set[0])
    ground_truth = query_set[1].numpy()

    # go through predictions query_size x ways, compare max softmax to ground truth, if it matches add sogftmax output to accumulating average of this class, otherwise add 0
    # TODO: what if np.max not unique
    for idx, prediction in enumerate(predictions): 
        label = ground_truth[idx]

        if np.argmax(prediction) == label: 
            conf_sc = np.max(prediction)
        else: 
            conf_sc = 0.0    

        if conf_scores[label]: 
            conf_scores[label] = np.mean(conf_scores[label], conf_sc)
        else: 
            conf_scores[label] = conf_sc

    return conf_scores