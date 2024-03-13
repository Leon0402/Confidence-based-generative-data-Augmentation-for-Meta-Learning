__all__ = ["ImageDataset"]

import pandas as pd
import numpy as np
import torch.utils.data
import torchvision.transforms
import torchvision.transforms.functional
import PIL

import cdmetadl.samplers

from .task import SetData, Task


class ImageDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for handling image datasets from MetaAlbum in PyTorch. 

    Attributes:
        img_paths (List[Path]): Paths to each image in the dataset.
        labels (numpy.ndarray): Encoded labels for each image.
        idx_per_label (List[numpy.ndarray]): Indices of images for each label.
        min_examples_per_class (int): The minimum number of examples across classes.
        transform (torchvision.transforms.Compose): Transformation to apply to each image.
    """

    def __init__(self, name: str, dataset_info: tuple, img_size: int = 128, included_classes: set[str] = None, offset: int = 0):
        """
        Args:
            name (str): Name of the dataset
            dataset_info (tuple): Contains the following entries:
                             - 'label_column': Column name for labels in metadata CSV.
                             - 'file_column': Column name for file names in metadata CSV.
                             - 'imgage_path': Path to the directory containing images.
                             - 'metadata_path': Path to the CSV file containing metadata.
            img_size (int, optional): Size to resize images to. Default is 128.
            offset (int): Keeps track of the number of labels that has already been assigned.
                        Relevant for avoiding duplicate labels in the batch-mode.
            included_classes (set[str]): Only includes datapoints with a ground truth label present in this set
        """
        self.name = name
        self.dataset_info = dataset_info
        self.img_size = img_size
        self.offset = offset
        label_column, file_column, imgage_path, metadata_path = dataset_info

        metadata = pd.read_csv(metadata_path)

        self.label_names = set(metadata[label_column])
        if included_classes is not None:
            self.label_names &= included_classes
        self.number_of_classes = len(self.label_names)

        self.text_to_numerical_label = {label: idx + offset for idx, label in enumerate(self.label_names)}
        self.numerical_label_to_text = {number: text for text, number in self.text_to_numerical_label.items()}
        self.labels = np.array([
            self.text_to_numerical_label[label] for label in metadata[label_column] if label in self.label_names
        ])

        self.img_paths = [
            imgage_path / name
            for name, label in metadata[[file_column, label_column]].values
            if label in self.label_names
        ]

        self.idx_per_label = [np.flatnonzero(self.labels == label) for label in self.text_to_numerical_label.values()]
        self.min_examples_per_class = min(len(idx) for idx in self.idx_per_label)
    
    def __len__(self) -> int:
        """
        Returns the number of images in the dataset.

        Returns:
            int: Total number of images.
        """
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the image and its label at a given index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the image and its label, both as tensors.
        """
        return self.read_image(idx), torch.tensor(self.labels[idx])

    def read_image(self, idx: int) -> torch.Tensor:
        return torchvision.io.read_image(str(self.img_paths[idx])) / 255

    def generate_task(
        self, n_ways: cdmetadl.samplers.Sampler, k_shots: cdmetadl.samplers.Sampler, query_size: int
    ) -> Task:
        if n_ways.max_value > self.number_of_classes:
            raise ValueError(
                f"Max ways was set to {n_ways.max_value}, but dataset {self.name} only has {self.number_of_classes} classes"
            )
        if k_shots.max_value + query_size > self.min_examples_per_class:
            raise ValueError(
                f"Max shots was set to {k_shots.max_value} and query size to {query_size}, but dataset {self.name} only has {self.min_examples_per_class} examples of its smallest class"
            )

        n_way = n_ways.sample()
        k_shot = k_shots.sample()

        selected_classes = np.random.permutation(self.number_of_classes)[:n_way]
        # Indices for support and query set are sampled together to ensure no indix appears twice
        sampled_indices_per_class = [
            np.random.choice(self.idx_per_label[cls], k_shot + query_size, replace=False) for cls in selected_classes
        ]

        support_set = SetData(
            images=torch.stack([
                self.read_image(idx) for indices in sampled_indices_per_class for idx in indices[:k_shot]
            ]),
            labels=torch.tensor(np.arange(n_way).repeat(k_shot)),
            number_of_ways=n_way,
            number_of_shots=k_shot,
            class_names=[self.numerical_label_to_text[idx + self.offset] for idx in selected_classes],
        )
        query_set = SetData(
            images=torch.stack([
                self.read_image(idx) for indices in sampled_indices_per_class for idx in indices[k_shot:]
            ]),
            labels=torch.tensor(np.arange(n_way).repeat(query_size)),
            number_of_ways=n_way,
            number_of_shots=k_shot,
            class_names=[self.numerical_label_to_text[idx + self.offset] for idx in selected_classes],
        )

        return Task(
            dataset_name=self.name,
            support_set=support_set,
            query_set=query_set,
            number_of_ways=n_way,
            class_names=[self.numerical_label_to_text[idx + self.offset] for idx in selected_classes],
        )
