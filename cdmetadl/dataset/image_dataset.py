__all__ = ["ImageDataset"]

import pandas as pd
import numpy as np
import torch.utils.data
import torchvision.transforms
import PIL

from .task import Task


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

    def __init__(self, name: str, dataset_info: dict, img_size: int = 128, included_classes: set[str] = None):
        """
        Args:
            name (str): Name of the dataset
            dataset_info (dict): Contains the following keys:
                             - 'label_column': Column name for labels in metadata CSV.
                             - 'file_column': Column name for file names in metadata CSV.
                             - 'imgage_path': Path to the directory containing images.
                             - 'metadata_path': Path to the CSV file containing metadata.
            img_size (int, optional): Size to resize images to. Default is 128.
            included_classes (set[str]): Only includes datapoints with a ground truth label present in this set
        """
        self.name = name
        self.dataset_info = dataset_info
        self.img_size = img_size
        label_column, file_column, imgage_path, metadata_path = dataset_info

        metadata = pd.read_csv(metadata_path)

        self.label_names = set(metadata[label_column])
        if included_classes is not None:
            self.label_names &= included_classes
        self.number_of_classes = len(self.label_names)

        self.text_to_numerical_label = {label: idx for idx, label in enumerate(self.label_names)}
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

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ToTensor()
        ])

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
        return self.transform(PIL.Image.open(self.img_pathsaaa[idx])), torch.tensor(self.labels[idx])

    def generate_task(self, min_ways: int, max_ways: int, min_shots: int, max_shots: int, query_size: int) -> Task:
        if max_ways > self.number_of_classes:
            raise ValueError(
                f"Max ways was set to {max_ways}, but dataset {self.name} only has {self.number_of_classes} classes"
            )
        if max_shots + query_size > self.min_examples_per_class:
            raise ValueError(
                f"Max shots was set to {max_shots} and query size to {query_size}, but dataset {self.name} only has {self.min_examples_per_class} examples of its smallest class"
            )

        n_way = np.random.randint(min_ways, max_ways + 1)
        k_shot = np.random.randint(min_shots, max_shots + 1)

        selected_classes = np.random.permutation(self.number_of_classes)[:n_way]
        # Indices for support and query set are sampled together to ensure no indix appears twice
        sampled_indices_per_class = [
            np.random.choice(self.idx_per_label[cls], k_shot + query_size, replace=False) for cls in selected_classes
        ]

        support_images = torch.stack([
            self[idx][0] for indices in sampled_indices_per_class for idx in indices[:k_shot]
        ])
        query_images = torch.stack([self[idx][0] for indices in sampled_indices_per_class for idx in indices[k_shot:]])

        return Task(
            num_ways=n_way,
            num_shots=k_shot,
            support_set=(
                support_images, torch.tensor(np.arange(n_way).repeat(k_shot)),
                torch.tensor(selected_classes.repeat(k_shot))
            ),
            query_set=(
                query_images, torch.tensor(np.arange(n_way).repeat(query_size)),
                torch.tensor(selected_classes.repeat(query_size))
            ),
            original_class_idx=selected_classes,
            dataset=self.name
        )
