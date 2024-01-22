import pathlib
import csv
from collections import defaultdict
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import cdmetadl.dataset
import cdmetadl.helpers.scoring_helpers


class TensorboardLogger():

    def __init__(self, tensorboard_dir: pathlib.Path):
        """
        Args:
            tensorboard_dir (pathlib.Path): Where the data for the tensorboard should be stored     
        """
        self.writer_train = SummaryWriter(tensorboard_dir / "train")
        self.writer_valid = SummaryWriter(tensorboard_dir / "valid")
        self.losses_train = []
        self.losses_valid = []
        self.scores_train = defaultdict(list)
        self.scores_valid = defaultdict(list)

    def _update_data(self, meta_train: bool, loss: float, scores: dict) -> None:
        """
        Update the tensorboard data (losses and scores).

        Args:
            meta_train (bool): Indicates if the update is for meta-train or meta-validation.
            loss (float): Loss value.
            scores (dict): Dictionary of scores.
        """

        if meta_train:
            self.losses_train.append(round(float(loss), 5))
            scores_dict = self.scores_train
        else:
            self.losses_valid.append(round(float(loss), 5))
            scores_dict = self.scores_valid

        for key in scores:
            scores_dict[key].append(round(scores[key], 5))

    def _avg_last_n_iters(self, losses: list, scores_dict: dict, n_average: int) -> tuple[float, dict[str, float]]:
        """
        Calculate the average loss and scores over the last n iterations.

        Args:
            losses (list): List of losses.
            scores_dict (dict): Dictionary of scores.
            n_average (int): Number of iterations to average over.

        Returns:
            Tuple[float, Dict[str, float]]: Average loss and scores.
        """
        loss_avg = np.average(losses[-n_average:])
        scores_avg = dict()

        for key in scores_dict:
            scores_avg[key] = np.average(scores_dict[key][-n_average:])

        return loss_avg, scores_avg

    def _write_log(self, writer: SummaryWriter, loss: float, scores: dict, meta_train_iteration: int):
        """
        Write loss and scores to Tensorboard.

        Args:
            writer: Tensorboard SummaryWriter.
            loss (float): Loss value.
            scores (dict): Dictionary of scores.
            meta_train_iteration (int): Meta-train iteration.
        """
        writer.add_scalar(f"Loss/step", loss, meta_train_iteration)

        for metric, value in scores.items():
            writer.add_scalar(f"Metrics/{metric}", value, meta_train_iteration)

    def _create_figure(
        self, title: str, data: np.array, labels: np.array, num_ways: int, sample_size: int, predictions=None
    ):
        """
        Create and save figures for Tensorboard.

        Args:
            title (str): Title of the figure.
            data (np.array): Input data.
            labels (np.array): Labels for the data.
            num_ways (int): Number of ways in the task.
            sample_size (int): Size of the sample / Number of shots
            predictions (Optional): Model predictions.
        """
        fig_support, axs_support = plt.subplots(
            sample_size, num_ways, figsize=(num_ways * 2, sample_size), squeeze=False, layout="constrained"
        )
        fig_support.suptitle(title)

        label_to_indices = {label: np.flatnonzero(labels == label) for label in np.unique(labels)}
        for label, indices in label_to_indices.items():
            axs_support[0, label].set_title(f"Class: {label}")
            for i, idx in enumerate(np.random.choice(indices, sample_size, replace=False)):
                axs_support[i, label].imshow(np.transpose(data[idx], (1, 2, 0)), cmap="gray")
                axs_support[i, label].axis('off')

                if predictions is not None:
                    probs = np.exp(predictions[idx]) / np.sum(np.exp(predictions[idx]))
                    sorted_indexes = np.argsort(probs)[::-1][:5]
                    pred_text = "\n".join([f"{idx}: {int(probs[idx]*100)}%" for idx in sorted_indexes])

                    axs_support[i, label].text(
                        1.05, 1, pred_text, transform=axs_support[i, label].transAxes, ha='left', va='top',
                        fontsize='medium', rotation='horizontal'
                    )

    def _write_samples(
        self, writer: SummaryWriter, data: cdmetadl.dataset.Task, predictions: np.array, meta_train_iteration: int
    ):
        """
        Write images/samples to Tensorboard.

        Args:
            writer (SummaryWriter): Tensorboard SummaryWriter.
            data (cdmetadl.dataset.Task): Task data.
            predictions (np.array): Model predictions.
            meta_train_iteration (int): Meta-train iteration.
        """
        self._create_figure(
            title="Support-Set", data=data.support_set.images.cpu().numpy(), labels=data.support_set.labels.cpu().numpy(),
            num_ways=data.number_of_ways, sample_size=min(data.support_set.number_of_shots, 5)
        )
        writer.add_figure(f"Dataset/Support Set", plt.gcf(), meta_train_iteration)

        # TODO: Adjustments needed for new task structure, possible in other places too
        self._create_figure(
            title="Query-Set", data=data.query_set.images.cpu().numpy(), labels=data.query_set.labels.cpu().numpy(),
            num_ways=data.number_of_ways, sample_size=min(data.query_set.number_of_shots, 5), predictions=predictions
        )
        writer.add_figure(f"Dataset/Query Set ", plt.gcf(), meta_train_iteration)

    def log(
        self, data: Any, scores: dict, loss: float, meta_train: bool, predictions: np.ndarray, val_tasks: int,
        val_after: int, meta_train_iteration: int, meta_validation_iteration: int, number_of_valid_datasets: int
    ) -> None:
        """
        Perform one loggin step and write current data to Tensorboard.

        Args:
            data (Any): Data used to compute predictions, it can be a task or a batch.
            scores (dict): Dictionary of scores.
            loss (float): Loss value.
            meta_train (bool): Boolean flag indicating if it's meta-train or meta-validation.
            predictions (np.ndarray): Model predictions.
            val_tasks (int): Total number of tasks used during validation.
            val_after (int): Number of training iterations before entering the validation stage.
            meta_train_iteration (int): Meta-train iteration.
            meta_validation_iteration (int): Meta-validation iteration.
            number_of_valid_datasets (int): Number of datasets used for meta-validation.
        """
        self._update_data(meta_train, loss, scores)

        training_epoch_done = meta_train_iteration % val_after == 0 and meta_train_iteration > 0
        validation_epoch_done = meta_validation_iteration % (
            val_tasks * number_of_valid_datasets
        ) == 0 and meta_validation_iteration > 0
        if training_epoch_done and validation_epoch_done:
            loss_train, scores_train = self._avg_last_n_iters(self.losses_train, self.scores_train, val_after)
            self._write_log(self.writer_train, loss_train, scores_train, meta_train_iteration)

            loss_valid, scores_valid = self._avg_last_n_iters(self.losses_valid, self.scores_valid, val_tasks)
            self._write_log(self.writer_valid, loss_valid, scores_valid, meta_train_iteration)

            self._write_samples(self.writer_valid, data, predictions, meta_train_iteration)


class Logger():
    """ Class to define the logger that can be used by the participants during 
    meta-learning to store detailed logs and get printed messages available in 
    the ingestion output log of the Competition Site. 
    """

    def __init__(
        self, logs_dir: pathlib.Path, tensorboard_dir: pathlib.Path | None, number_of_valid_datasets: int
    ) -> None:
        """
        Args:
            logs_dir (pathlib.Path): Directory where the logs should be stored.

            tensorboard_dir (pathlib.Path | None): Indicates if and where the data
                    for the tensorboard should be stored     

            number_of_valid_datasets (int): Number of datasets that will be used for meta-validation
        """
        self.logs_dir = logs_dir
        self.meta_train_iteration = 0
        self.meta_train_logs_path = self.logs_dir / "meta_train"
        self.meta_validation_iteration = 0
        self.meta_valid_step = 0
        self.meta_valid_root_path = self.logs_dir / "meta_validation"
        self.print_separator = False
        self.number_of_valid_datasets = number_of_valid_datasets
        if tensorboard_dir is not None:
            self.tensorboard_logger = TensorboardLogger(tensorboard_dir)

    def log(
        self, data: Any, predictions: np.ndarray, loss: float, val_tasks: int, val_after: int, meta_train: bool = True
    ) -> None:
        """ Store the task/batch information, predictions, loss and scores of 
        the current meta-train or meta-validation iteration.

        Args:
            data (Any): Data used to compute predictions, it can be a task or a 
                batch.

            predictions (np.ndarray): Predictions associated to each test 
                example in the specified data. It can be the raw logits matrix 
                (the logits are the unnormalized final scores of your model), 
                a probability matrix, or the predicted labels.

            loss (float): Loss of the current iteration.

            val_tasks (int): The total number of tasks that will be used
                    during the validation stage.

            val_after (int): The number of training iterations that will be completed
                    before entering the validation stage.umber of traing iterations that
                    will be performed between the validation stage.
            
            meta_train (bool, optional): Boolean flag to control if the current 
                iteration belongs to meta-training. Defaults to True.

        """
        is_task = isinstance(data, cdmetadl.dataset.Task)

        first_log = False
        if meta_train:
            # Create log dirs
            if self.meta_train_iteration == 0:
                first_log = True
                self._create_logs_dirs(self.meta_train_logs_path)

            # Print separator after finishing meta-valid step
            if self.print_separator:
                self.meta_validation_iteration = 0
                self.print_separator = False
                print(f"{'#'*79}\n")

            # Prepare paths to files
            self.meta_train_iteration += 1
            ground_truth_path = f"{self.meta_train_logs_path}/ground_truth"
            predictions_path = f"{self.meta_train_logs_path}/predictions"
            task_file = f"{self.meta_train_logs_path}/tasks.csv"
            performance_file = f"{self.meta_train_logs_path}/performance.csv"
            curr_iter = f"iteration_{self.meta_train_iteration}.out"
            print_text = f"Meta-train iteration {self.meta_train_iteration:>4d}:"

        else:
            # Create log dirs
            if self.meta_validation_iteration == 0:
                first_log = True
                self.meta_valid_step += 1
                self.meta_valid_logs_path = self.meta_valid_root_path / f"step_{self.meta_valid_step}"
                self.meta_valid_logs_path.mkdir(parents=True)
                print(f"\n{'#'*30} Meta-valid step {self.meta_valid_step} " + f"{'#'*30}")
                self.print_separator = True

            self.meta_validation_iteration += 1
            task_file = f"{self.meta_valid_logs_path}/tasks.csv"
            performance_file = f"{self.meta_valid_logs_path}/performance.csv"
            print_text = f"Meta-valid iteration {self.meta_validation_iteration:>4d}:"

        if is_task:
            # Save task information
            dataset = data.dataset_name
            N = data.number_of_ways
            k = data.query_set.number_of_shots

            with open(task_file, "a", newline="") as f:
                writer = csv.writer(f)
                if first_log:
                    writer.writerow(["Dataset", "N", "k"])
                writer.writerow([dataset, N, k])

            ground_truth = data.query_set.labels.cpu().numpy()

        else:
            N = None
            ground_truth = data[1].cpu().numpy()

        if meta_train:
            # Save ground truth and predicted values
            np.savetxt(f"{ground_truth_path}/{curr_iter}", ground_truth, fmt="%d")
            fmt = "%f" if len(predictions.shape) == 2 else "%d"
            np.savetxt(f"{predictions_path}/{curr_iter}", predictions, fmt=fmt)

        # Compute and save performance
        scores = cdmetadl.helpers.scoring_helpers.compute_all_scores(ground_truth, predictions, N, not is_task)

        if is_task:
            print(
                f"{print_text}\t{scores['Normalized Accuracy']:.4f} (Normalized Accuracy)\t{scores['Accuracy']:.4f} (Accuracy)\t{loss:.4f} (Loss)\t[{N}-way {k}-shot task from {dataset}]"
            )
        else:
            print(f"{print_text}\t{scores['Accuracy']:.4f} (Accuracy)\t{loss:.4f} (Loss)")

        if self.tensorboard_logger is not None:
            self.tensorboard_logger.log(
                data, scores, loss, meta_train, predictions, val_tasks, val_after, self.meta_train_iteration,
                self.meta_validation_iteration, self.number_of_valid_datasets
            )

        scores["loss"] = loss
        with open(performance_file, "a", newline="") as f:
            writer = csv.writer(f)
            if first_log:
                writer.writerow(list(scores.keys()))
            writer.writerow(list(scores.values()))

    def _create_logs_dirs(self, dir: pathlib.Path) -> None:
        """ Create all the necessary directories for storing the logs at 
        meta-training time.

        Args:
            dir (str): Directory where the log directories should be created.
        """
        for value_to_log in ["ground_truth", "predictions"]:
            log_dir = dir / value_to_log
            log_dir.mkdir(parents=True)
