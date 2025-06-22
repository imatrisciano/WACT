import torch
from joblib import Parallel, delayed
from torch.utils.data import Dataset

from src.DataPreprocessing.ChangeDetector import ChangeDetector
from src.DataPreprocessing.ObjectEncoder import ObjectEncoder
from src.DataPreprocessing.WorldStatusEncoder import MostChangesWorldStatusEncoder
from src.ObjectStore.MetadataObjectStore import MetadataObjectStore


class MyDataset(Dataset):
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
            label = self.label_to_id_map.get(obj["action_name"], 0)

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]