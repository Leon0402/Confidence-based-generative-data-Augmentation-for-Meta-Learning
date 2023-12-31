__all__ = ["MetaLearner", "Learner", "Predictor"]
"""API for models
"""
import numpy as np

import cdmetadl.config
import cdmetadl.dataset


class Predictor():
    """ This class represents the predictor returned at the end of the 
    Learner's fit method. 
    """

    def __init__(self) -> None:
        """ Defines the Predictor initialization.
        """
        pass

    def predict(self, query_set: cdmetadl.dataset.SetData) -> np.ndarray:
        """ Given a query_set, predicts the probabilities associated to the 
        provided images or the labels to the provided images.
        
        Args:
            query_set (cdmetadl.dataset.SetData): Query set without labels for prediction.
        
        Returns:
            np.ndarray: It can be:
                - Raw logits matrix (the logits are the unnormalized final 
                    scores of your model). The matrix must be of shape 
                    [n_ways*query_size, n_ways]. 
                - Predicted label probabilities matrix. The matrix must be of 
                    shape [n_ways*query_size, n_ways].
                - Predicted labels. The array must be of shape 
                    [n_ways*query_size].
        """
        raise NotImplementedError(("You should implement the predict method " + "for the Predictor class."))


class Learner():
    """ This class represents the learner returned at the end of the 
    meta-learning procedure.
    """

    def __init__(self) -> None:
        """ Defines the learner initialization.
        """
        pass

    def fit(self, support_set: cdmetadl.dataset.SetData) -> Predictor:
        """ Fit the Learner to the support set of a new unseen task. 
        
        Args:
            support_set (cdmetadl.dataset.SetData): Support set used for finetuning the learner.
                        
        Returns:
            Predictor: The resulting predictor ready to predict unlabelled 
                query image examples from new unseen tasks.
        """
        raise NotImplementedError(("You should implement the fit method for " + "the Learner class."))

    def save(self, path_to_save) -> None:
        """ Saves the learning object associated to the Learner. 
        
        Args:
            path_to_save (str): Path where the learning object will be saved.

        Note: It is mandatory to allow saving the Learner as a file(s) in 
        path_to_save. Otherwise, it won't be a valid submission.
        """
        raise NotImplementedError(("You should implement the save method for " + "the Learner class."))

    def load(self, path_to_load) -> None:
        """ Loads the learning object associated to the Learner. It should 
        match the way you saved this object in self.save().
        
        Args:
            path_to_load (str): Path where the Learner is saved.
        """
        raise NotImplementedError(("You should implement the load method for " + "the Learner class."))


class MetaLearner():
    """ Define the meta-learning algorithm we want to use, through its methods.
    It is an abstract class so one has to overide the core methods depending 
    on the algorithm.
    """
    """
    Which data formats the meta learner expects 
    """
    data_format = cdmetadl.config.DataFormat.TASK

    def __init__(self, train_classes, total_classes, logger, mode=cdmetadl.config.DataFormat.TASK) -> None:
        """ 
        Defines the meta-learning algorithm's parameters. For example, one 
        has to define what would be the meta-learner's architecture. 
        
        Args:
            train_classes (int): Total number of classes that can be seen 
                during meta-training. If the data format during training is 
                'task', then this parameter corresponds to the number of ways, 
                while if the data format is 'batch', this parameter corresponds 
                to the total number of classes across all training datasets.
            total_classes (int): Total number of classes across all training 
                datasets. If the data format during training is 'batch' this 
                parameter is exactly the same as train_classes.
            logger (Logger): Logger that you can use during meta-learning (HIGHLY RECOMMENDED). 
                You can use it after each meta-train or meta-validation iteration as follows: 
                    self.log(data, predictions, loss, meta_train)
                - data (task or batch): It is the data used in the current 
                    iteration.
                - predictions (np.ndarray): Predictions associated to each test 
                    example in the specified data. It can be the raw logits 
                    matrix (the logits are the unnormalized final scores of 
                    your model), a probability matrix, or the predicted labels.
                - loss (float, optional): Loss of the current iteration. 
                    Defaults to None.
                - meta_train (bool, optional): Boolean flag to control if the 
                    current iteration belongs to meta-training. Defaults to 
                    True.  
        """
        self.train_classes = train_classes
        self.total_classes = total_classes
        self.log = logger.log

    def meta_fit(
        self, meta_train_generator: cdmetadl.dataset.DataGenerator, meta_valid_generator: cdmetadl.dataset.TaskGenerator
    ) -> Learner:
        """ 
        Uses the generators to tune the meta-learner's parameters. The meta-training generator generates either few-shot 
        learning tasks or batches of images, while the meta-valid generator always generates few-shot learning tasks.
        
        Args:
            meta_train_generator (cdmetadl.dataset.DataGenerator):  Generator for train data in batch or task format
            meta_valid_generator (cdmetadl.dataset.TaskGenerator): Generator for validation data in task format
                
        Returns:
            Learner: Resulting learner ready to be trained and evaluated on new unseen tasks.
        """
        raise NotImplementedError(("You should implement the meta_fit method " + f"for the MetaLearner class."))
