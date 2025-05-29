import numpy as np


class ObjectEncoder:
    def __init__(self, object_embedding_size:int):
        self.object_embedding_size = object_embedding_size
        pass

    def encode(self, obj: dict) -> np.array:
        if obj is None:
            return np.zeros(self.object_embedding_size)

        raise NotImplementedError

    def decode(self, obj: np.array) -> dict:
        raise NotImplementedError
