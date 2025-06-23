from typing import Sized

import numpy as np
import torch
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from src.DataPreprocessing.ChangeDetector import ChangeDetector
from src.DataPreprocessing.ObjectEncoder import ObjectEncoder
from src.DataPreprocessing.WorldStatusEncoder import MostChangesWorldStatusEncoder
from src.ObjectStore.MetadataObjectStore import MetadataObjectStore


class MyDataset(Dataset, Sized):
    """
    A simple synthetic dataset for demonstration purposes.
    Generates random float vectors and corresponding labels.
    """
    def __init__(self, object_store: MetadataObjectStore, number_of_significant_objects: int = 10):
        self.object_store = object_store
        self.change_detector = ChangeDetector()

        self.object_encoder = ObjectEncoder()
        self.world_status_encoder = MostChangesWorldStatusEncoder(self.object_encoder, number_of_significant_objects)

        self.dataset_files = self.object_store.list_files()
        self.id_to_labels_map = {
            0: "Unknown",
            1: "Pickup Object",
            2: "Cook Object",
            3: "Slice Object",
            4: "Fill Object",
            5: "Toggle Off Object",
            6: "Open Object",
            7: "Toggle On Object",
            8: "Break Object",
            9: "Dirty Object",
            10: "Empty Object",
            11: "Close Object",
            12: "Clean Object",
        }
        self.label_to_id_map = {v: k for k, v in self.id_to_labels_map.items()}

        def _preprocess_item(item_path) -> tuple:
            """
            Reads an action file from disk and gathers its data and label
            :param item_path: action file's path on disk
            :return: tuple (data, label)
            """
            obj = self.object_store.load(item_path)

            data = self.world_status_encoder.encode_action_data(obj)
            label = self.label2id(obj["action_name"])

            return data, label

        print("Preprocessing dataset...")
        # Preprocess all input files in a parallel manner
        results = (Parallel(n_jobs=-1)
                   (delayed(_preprocess_item)(path)
                    for path in self.dataset_files))

        # Results is a list of tuples [(data_1, label_1), (data_2, label_2), ..., (data_n, label_n)],
        # We turn it into a list of data and a list of labels
        self.data, self.labels = zip(*results)

        # Now let's make the data into pytorch's tensors, ready to be moved to the correct device
        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        print("Dataset ready")

    def label2id(self, label: str) -> int:
        return self.label_to_id_map.get(label, 0)

    def id2label(self, index: int) -> str:
        return self.id_to_labels_map.get(index, "Unknown")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def split_dataset(
            self,
            train_split_ratio: float,
            val_split_ratio: float,
            batch_size: int,
            shuffle_dataset: bool = True) -> (DataLoader, DataLoader, DataLoader):
        """
           Splits a dataset into training, validation, and test sets.

           Args:
               train_split_ratio (float): The proportion of the dataset to allocate to the training set.
                                          Must be between 0 and 1.
               val_split_ratio (float): The proportion of the dataset to allocate to the validation set.
                                        Must be between 0 and 1.
                                        The test set will take the remaining proportion.
               batch_size (int): The batch size, so DataLoaders can properly be defined
               shuffle_dataset (bool): Whether to shuffle the dataset indices before splitting.

           Returns:
               tuple: A tuple containing (train_loader, validation_loader, test_loader).
           """

        whole_dataset = self

        if not (0 < train_split_ratio < 1 and 0 <= val_split_ratio < 1):
            raise ValueError("train_split_ratio and val_split_ratio must be between 0 and 1.")
        if train_split_ratio + val_split_ratio >= 1:
            raise ValueError("The sum of train_split_ratio and val_split_ratio must be less than 1 "
                             "to leave room for a test set.")

        dataset_size = len(whole_dataset)
        indices = list(range(dataset_size))

        if shuffle_dataset:
            np.random.shuffle(indices)

        # Calculate split points
        train_split_point = int(np.floor(train_split_ratio * dataset_size))
        val_split_point = int(np.floor((train_split_ratio + val_split_ratio) * dataset_size))

        # Split indices
        train_indices = indices[:train_split_point]
        val_indices = indices[train_split_point:val_split_point]
        test_indices = indices[val_split_point:]

        # Create samplers
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(whole_dataset, batch_size=batch_size, sampler=train_sampler)
        validation_loader = DataLoader(whole_dataset, batch_size=batch_size, sampler=valid_sampler)
        test_loader = DataLoader(whole_dataset, batch_size=batch_size, sampler=test_sampler)

        return train_loader, validation_loader, test_loader