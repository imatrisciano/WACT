import os.path
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
    def __init__(self, object_store: MetadataObjectStore, use_cache:bool = True, cache_location: str = "../data/dataset_cache/", number_of_significant_objects: int = 10, use_parallel_processing: bool = True):
        self.action_labels = None
        self.object_labels = None
        self.data = None
        self.loaded = False
        self.use_cache: bool = use_cache
        self.cache_location: str = cache_location
        self.use_parallel_processing: bool = use_parallel_processing

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
            13: "Put Object"
        }
        self.label_to_id_map = {v: k for k, v in self.id_to_labels_map.items()}

    def _process_raw_dataset(self):
        def _preprocess_item(item_path) -> tuple:
            """
            Reads an action file from disk and gathers its data and label
            :param item_path: action file's path on disk
            :return: tuple (data, label)
            """
            obj = self.object_store.load(item_path)

            data, object_label = self.world_status_encoder.encode_action_data_and_return_action_object_index(obj)
            action_label = self.label2id(obj["action_name"])

            return data, action_label, object_label

        print("Preprocessing dataset...")

        if self.use_parallel_processing:
            # Preprocess all input files in a parallel manner
            results = (Parallel(n_jobs=-1)
                      (delayed(_preprocess_item)(path)
                       for path in self.dataset_files))
        else:
            # Preprocess all input files single threaded
            results = []
            for path in self.dataset_files:
                results.append(_preprocess_item(path))

        # Results is a list of tuples [(data_1, action_label_1, object_label_1),
        #  (data_2, action_label_2, object_label_2), ..., (data_n, action_label_n, object_label_n)],
        # We turn it into a list of data and a list of labels
        self.data, self.action_labels, self.object_labels = zip(*results)

        # Now let's make the data into pytorch's tensors, ready to be moved to the correct device
        self.data = torch.tensor(self.data)
        self.action_labels = torch.tensor(self.action_labels, dtype=torch.long)
        self.object_labels = torch.tensor(self.object_labels, dtype=torch.long)

        print("Dataset ready")

        if self.use_cache:
            self._save_dataset_cache()

    def _load_cached_dataset(self):
        print(f"Loading cached dataset from location: {self.cache_location}")
        data_path, action_labels_path, object_labels_path = self._get_data_and_label_cache_path()
        self.data = torch.tensor(torch.load(data_path))
        self.action_labels = torch.tensor(torch.load(action_labels_path), dtype=torch.long)
        self.object_labels = torch.tensor(torch.load(object_labels_path), dtype=torch.long)

    def _save_dataset_cache(self):
        print(f"Saving dataset cache in location: {self.cache_location}")
        os.makedirs(self.cache_location)

        data_path, action_labels_path, object_labels_path = self._get_data_and_label_cache_path()
        torch.save(self.data, data_path)
        torch.save(self.action_labels, action_labels_path)
        torch.save(self.object_labels, object_labels_path)

    def _get_data_and_label_cache_path(self) -> (str, str):
        data_path = os.path.join(self.cache_location, "data.bin")
        action_labels_path = os.path.join(self.cache_location, "action_labels.bin")
        object_labels_path = os.path.join(self.cache_location, "object_labels.bin")

        return data_path, action_labels_path, object_labels_path

    def _cache_exists(self) -> bool:
        return all(os.path.exists(x) for x in self._get_data_and_label_cache_path())

    def load(self):
        if self.use_cache and self._cache_exists():
            self._load_cached_dataset()
        else:
            self._process_raw_dataset()

        self.loaded = True

    def label2id(self, label: str) -> int:
        return self.label_to_id_map.get(label, 0)

    def id2label(self, index: int) -> str:
        return self.id_to_labels_map.get(index, "Unknown")

    def get_labels(self) -> list[str]:
        return list(self.id_to_labels_map.values())

    def __len__(self):
        if not self.loaded:
            raise Exception("Dataset must be loaded. Please invoke `load()` before continuing")
        return len(self.data)

    def __getitem__(self, idx):
        if not self.loaded:
            raise Exception("Dataset must be loaded. Please invoke `load()` before continuing")

        return self.data[idx], self.action_labels[idx], self.object_labels[idx]

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
        if not self.loaded:
            raise Exception("Dataset must be loaded. Please invoke `load()` before continuing")
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