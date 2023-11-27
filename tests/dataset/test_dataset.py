import unittest
import pathlib

import torch

import cdmetadl.helpers.general_helpers
import cdmetadl.dataset.split

DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "public_data"


class TestImageDataset(unittest.TestCase):

    def setUp(self):
        # Load dataset information
        self.train_datasets_info, self.valid_datasets_info, self.test_datasets_info = \
            cdmetadl.helpers.general_helpers.prepare_datasets_information(
                DATA_DIR, 2, 42, False)

    def test_init(self):
        # Test initialization with train dataset info
        for dataset_name, dataset_info in self.train_datasets_info.items():
            dataset = cdmetadl.dataset.split.ImageDataset(dataset_info)
            self.assertIsInstance(
                dataset, cdmetadl.dataset.split.ImageDataset, f"Failed to initialize ImageDataset for {dataset_name}"
            )

    def test_length(self):
        # Test the length of the dataset
        for dataset_name, dataset_info in self.train_datasets_info.items():
            dataset = cdmetadl.dataset.split.ImageDataset(dataset_info)
            self.assertTrue(len(dataset) > 0, f"Dataset {dataset_name} is empty.")

    def test_getitem(self):
        # Test retrieving an item
        for dataset_name, dataset_info in self.train_datasets_info.items():
            dataset = cdmetadl.dataset.split.ImageDataset(dataset_info)
            image, label = dataset[0]  # Retrieve first item
            self.assertIsInstance(image, torch.Tensor, f"Image is not a torch.Tensor for {dataset_name}")
            self.assertIsInstance(label, torch.Tensor, f"Label is not a torch.Tensor for {dataset_name}")

    def test_getitem_invalid_index(self):
        # Test retrieving an item with an invalid index
        for dataset_name, dataset_info in self.train_datasets_info.items():
            dataset = cdmetadl.dataset.split.ImageDataset(dataset_info)
            with self.assertRaises(IndexError):
                _ = dataset[len(dataset)]


if __name__ == '__main__':
    unittest.main()
