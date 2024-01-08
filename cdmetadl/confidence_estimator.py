import torch
import pandas as pd

import cdmetadl.dataset
import cdmetadl.helpers.general_helpers
import cdmetadl.helpers.scoring_helpers
import cdmetadl.dataset.split



# should be in its own file
def ref_set_confidence_scores(conf_support_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], conf_query_set: tuple[torch.Tensor, torch.Tensor, torch.Tensor], learner: MyLearner) -> dict: 
    # for pseudo augmentation
    # returns dict with confidence scores for each class found in this task, calculated as 0 if prediciton is false or a mean of the softmaxes over all shots if prediction is true
    conf_scores = dict()
    learner.fit(support_set)
    predictions = predictor.predict(query_set[0])
    # need softmax and label of predicted item
    print("predictions: ", predictions)


    return conf_scores