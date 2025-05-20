import os
import json


class MetadataObjectStore:
    def __init__(self, base_path: os.PathLike | str):
        self.base_path: os.PathLike = base_path

    def store(self, filename: os.PathLike | str, obj: dict):
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

        file_path = os.path.join(self.base_path, filename)
        with open(file_path, "w") as file:
            json.dump(obj, file, indent=True)

    def load(self, filename: os.PathLike | str) -> dict:
        file_path = os.path.join(self.base_path, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError

        with open(file_path, "r") as file:
            obj = json.load(file)

        return obj
