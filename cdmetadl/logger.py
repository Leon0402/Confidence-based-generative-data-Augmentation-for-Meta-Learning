import pathlib
import torch
import csv
import io
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Union, Tuple, Dict

import cdmetadl.dataset
from cdmetadl.helpers.scoring_helpers import compute_all_scores
from collections import defaultdict
import matplotlib.pyplot as plt


class Logger():
    """ Class to define the logger that can be used by the participants during 
    meta-learning to store detailed logs and get printed messages available in 
    the ingestion output log of the Competition Site. 
    """

    def __init__(self, logs_dir: pathlib.Path, tensorboard_dir: Union[pathlib.Path, None], number_of_valid_datasets: int) -> None:
        """
        Args:
            logs_dir (pathlib.Path): Directory where the logs should be stored.

            tensorboard_dir Union[pathlib.Path, None]: Indicates if and where the data
                    for the tensorboard should be stored     

            number_of_valid_datasets (int): Number of datasets that will be used for meta-validation
        """
        self.logs_dir = logs_dir
        self.meta_train_iterations = 0
        self.meta_train_logs_path = self.logs_dir / "meta_train"
        self.meta_validation_iterations = 0
        self.meta_valid_steps = 0
        self.meta_valid_root_path = self.logs_dir / "meta_validation"
        self.print_separator = False
        self.use_tensorboard = False
        self.number_of_valid_datasets = number_of_valid_datasets

        if tensorboard_dir != None:
            self.writer_train = SummaryWriter(f"{tensorboard_dir}/train")
            self.writer_valid = SummaryWriter(f"{tensorboard_dir}/valid")
            self.use_tensorboard = True
            self.last_meta_train_state = True
            self.losses_train = []
            self.losses_valid = []
            self.scores_train = defaultdict(list)
            self.scores_valid = defaultdict(list)


    def _tensorboard_update_data(self, meta_train: bool, loss: float, scores: dict) -> None:
        if meta_train:
            self.losses_train.append(round(float(loss), 5))
            scores_dict = self.scores_train
        else:
            self.losses_valid.append(round(float(loss), 5))
            scores_dict = self.scores_valid
        
        for key in scores:
            scores_dict[key].append(round(scores[key], 5))
            
    def _tensorboard_avg_last_n_iters(self, losses: list, scores_dict: dict, n_average: int) -> Tuple[float, Dict[str, float]]:
        loss_avg = np.average(losses[-n_average:])
        scores_avg = dict()

        for key in scores_dict:
            scores_avg[key] = np.average(scores_dict[key][-n_average:])

        return loss_avg, scores_avg
            
    def _tensorboard_write_log(self, writer, loss, scores):
        writer.add_scalar(f"Loss/iteration", loss, self.meta_train_iterations )

        for metric, value in scores.items():
            writer.add_scalar(f"{metric}", value, self.meta_train_iterations )

    def _tensorboard_write_samples(self, writer, data, predictions, is_task):
        #TODO: Add a nice view for the samples, this is not it
        if is_task:
            predictions # array of size (100, 5)
            query_set_data = data.query_set[0].cpu().numpy() # array of shape (100, 3, 128, 128)
            query_set_labels = data.query_set[1].cpu().numpy() # array of shape (100,)


            for i in range(len(predictions)):
                prediction_i = predictions[i].round(3)
                query_data_i = query_set_data[i]
                query_labels_i = query_set_labels[i]

                # Example: Plot the first channel of the query_set_data
                plt.imshow(query_data_i[0, :, :], cmap='gray')
                plt.title(f"Prediction: {prediction_i}, Label: {query_labels_i}")
                plt.axis('off')

                # Save the figure to a BytesIO object
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)

                # Convert the figure to a NumPy array
                image_np = plt.imread(buf)

                # Convert the NumPy array to a PyTorch tensor
                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

                # Add the image to TensorBoard
                writer.add_image(f'Grid_Plot/{i}', image_tensor, global_step=i)

                # Close the figure to free up resources
                plt.close()


    def _tensorboard_log(self, data: Any, scores: dict, loss: float, meta_train: bool,
                        is_task: bool, predictions: np.ndarray, val_tasks: int, val_after: int) -> None:
        self._tensorboard_update_data(meta_train, loss, scores)
        
        if (self.meta_train_iterations % val_after == 0) and \
            (self.meta_validation_iterations % (val_tasks*self.number_of_valid_datasets) == 0) and\
            self.meta_train_iterations > 0 and self.meta_validation_iterations > 0:
            loss_train, scores_train = self._tensorboard_avg_last_n_iters(self.losses_train, self.scores_train, val_after)
            loss_valid, scores_valid = self._tensorboard_avg_last_n_iters(self.losses_valid, self.scores_valid, val_tasks)
            self._tensorboard_write_log(self.writer_train, loss_train, scores_train)
            self._tensorboard_write_log(self.writer_valid, loss_valid, scores_valid)

            #self._tensorboard_write_samples(self.writer_valid, data, predictions, is_task)
        self.last_meta_train_state = meta_train

    def log(self, data: Any, predictions: np.ndarray, loss: float, val_tasks:int, val_after: int, meta_train: bool = True) -> None:
        """ Store the task/batch information, predictions, loss and scores of 
        the current meta-train or meta-validation iteration.

        Args:
            data (Any): Data used to compute predictions, it can be a task or a 
                batch.

            predictions (np.ndarray): Predictions associated to each test 
                example in the specified data. It can be the raw logits matrix 
                (the logits are the unnormalized final scores of your model), 
                a probability matrix, or the predicted labels.

            loss (float, optional): Loss of the current iteration. Defaults to 
                None.

            val_tasks (int): The total number of tasks that will be used
                    during the validation stage.

            val_after (int): The number of training iterations that will be completed
                    before entering the validation stage.umber of traing iterations that
                    will be performed between the validation stage.
            
            meta_train (bool, optional): Boolean flag to control if the current 
                iteration belongs to meta-training. Defaults to True.

        """
        # Check the data format
        is_task = False
        if isinstance(data, cdmetadl.dataset.Task):
            is_task = True

        first_log = False
        if meta_train:
            # Create log dirs
            if self.meta_train_iterations == 0:
                first_log = True
                self._create_logs_dirs(self.meta_train_logs_path)

            # Print separator after finishing meta-valid step
            if self.print_separator:
                self.meta_validation_iterations = 0
                self.print_separator = False
                print(f"{'#'*79}\n")

            # Prepare paths to files
            self.meta_train_iterations += 1
            ground_truth_path = f"{self.meta_train_logs_path}/ground_truth"
            predictions_path = f"{self.meta_train_logs_path}/predictions"
            task_file = f"{self.meta_train_logs_path}/tasks.csv"
            performance_file = f"{self.meta_train_logs_path}/performance.csv"
            curr_iter = f"iteration_{self.meta_train_iterations}.out"
            print_text = f"Meta-train iteration {self.meta_train_iterations}:"

        else:
            # Create log dirs
            if self.meta_validation_iterations == 0:
                first_log = True
                self.meta_valid_steps += 1
                self.meta_valid_logs_path = self.meta_valid_root_path / f"step_{self.meta_valid_steps}"
                self.meta_valid_logs_path.mkdir(parents=True)
                print(f"\n{'#'*30} Meta-valid step {self.meta_valid_steps} " + f"{'#'*30}")
                self.print_separator = True

            self.meta_validation_iterations += 1
            task_file = f"{self.meta_valid_logs_path}/tasks.csv"
            performance_file = f"{self.meta_valid_logs_path}/performance.csv"
            print_text = "Meta-valid iteration " \
                + f"{self.meta_validation_iterations}:"

        if is_task:
            # Save task information
            dataset = data.dataset
            N = data.num_ways
            k = data.num_shots

            with open(task_file, "a", newline="") as f:
                writer = csv.writer(f)
                if first_log:
                    writer.writerow(["Dataset", "N", "k"])
                writer.writerow([dataset, N, k])


            ground_truth = data.query_set[1].cpu().numpy()

        else:
            N = None
            ground_truth = data[1].cpu().numpy()

        if meta_train:
            # Save ground truth and predicted values
            np.savetxt(f"{ground_truth_path}/{curr_iter}", ground_truth, fmt="%d")
            fmt = "%f" if len(predictions.shape) == 2 else "%d"
            np.savetxt(f"{predictions_path}/{curr_iter}", predictions, fmt=fmt)

        # Compute and save performance
        scores = compute_all_scores(ground_truth, predictions, N, not is_task)
        score_names = list(scores.keys())
        score_values = list(scores.values())


        if loss is not None:
            score_names.append("Loss")
            score_values.append(loss)
            if is_task:
                print(
                    f"{print_text}" + f"\t{scores['Normalized Accuracy']:.4f} (Normalized " + "Accuracy)" +
                    f"\t{scores['Accuracy']:.4f} (Accuracy)" + f"\t{loss:.4f} (Loss)" +
                    f"\t[{N}-way {k}-shot task from {dataset}]"
                )
            else:
                print(f"{print_text}" + f"\t{scores['Accuracy']:.4f} (Accuracy)" + f"\t{loss:.4f} (Loss)")
        else:
            if is_task:
                print(
                    f"{print_text}" + f"\t{scores['Normalized Accuracy']:.4f} (Normalized " + "Accuracy)" +
                    f"\t{scores['Accuracy']:.4f} (Accuracy)" + f"\t[{N}-way {k}-shot task from {dataset}]"
                )
            else:
                print(f"{print_text}" + f"\t{scores['Accuracy']:.4f} (Accuracy)")

        if self.use_tensorboard:
            self._tensorboard_log(data, scores, loss, meta_train, is_task, predictions, val_tasks, val_after)
        
        with open(performance_file, "a", newline="") as f:
            writer = csv.writer(f)
            if first_log:
                writer.writerow(score_names)
            writer.writerow(score_values)

    def _create_logs_dirs(self, dir: pathlib.Path) -> None:
        """ Create all the necessary directories for storing the logs at 
        meta-training time.

        Args:
            dir (str): Directory where the log directories should be created.
        """
        for value_to_log in ["ground_truth", "predictions"]:
            log_dir = dir / value_to_log
            log_dir.mkdir(parents=True)
