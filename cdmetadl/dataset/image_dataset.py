import random

import pandas as pd
import numpy as np
import torch.utils.data
import torchvision.transforms
import sklearn.utils
import PIL

import cdmetadl.dataset


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

    def __init__(self, dataset_info: dict, img_size: int = 128):
        """
        Args:
            dataset_info (dict): Contains the following keys:
                             - 'label_column': Column name for labels in metadata CSV.
                             - 'file_column': Column name for file names in metadata CSV.
                             - 'imgage_path': Path to the directory containing images.
                             - 'metadata_path': Path to the CSV file containing metadata.
            img_size (int, optional): Size to resize images to. Default is 128.
        """
        label_column, file_column, imgage_path, metadata_path = dataset_info

        metadata = pd.read_csv(metadata_path)

        self.img_paths = [imgage_path / image_name for image_name in metadata[file_column]]
        label_to_id = {label: id for id, label in enumerate(set(metadata[label_column]))}
        self.labels = np.array([label_to_id[label] for label in metadata[label_column]])

        self.idx_per_label = [np.flatnonzero(self.labels == i) for i in range(max(self.labels) + 1)]
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
        return self.transform(PIL.Image.open(self.img_paths[idx])), torch.tensor(self.labels[idx])

    def generate_task(self, min_ways: int, max_ways: int, min_shots: int, max_shots: int) -> cdmetadl.dataset.Task:
        generator = sklearn.utils.check_random_state(42)

        # TODO(leon): Needs more work with error handling (n_way > number of classes for instance)
        n_way = generator.randint(min_ways, max_ways + 1)
        k_shot = generator.randint(min_shots, max_shots + 1)
        query_size = 10

        selected_classes = generator.permutation(len(self.idx_per_label))[:n_way]

        # Initialize support and query sets
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        original_class_idx = []

        for cls in selected_classes:
            all_indices = self.idx_per_label[cls]
            sampled_indices = generator.choice(all_indices, k_shot + query_size, replace=False)
            support_idx, query_idx = sampled_indices[:k_shot], sampled_indices[k_shot:]

            # Add to support and query sets
            for idx in support_idx:
                img, label = self[idx]
                support_images.append(img)
                support_labels.append(torch.tensor(cls))

            for idx in query_idx:
                img, label = self[idx]
                query_images.append(img)
                query_labels.append(torch.tensor(cls))

            original_class_idx.append(cls)

        return cdmetadl.dataset.Task(
            num_ways=n_way,
            num_shots=k_shot,
            support_set=(
                torch.stack(support_images), torch.tensor(support_labels), torch.tensor(np.array(original_class_idx))
            ),
            query_set=(
                torch.stack(query_images), torch.tensor(query_labels), torch.tensor(np.array(original_class_idx))
            ),
            original_class_idx=original_class_idx,
            dataset=self.name
        )
