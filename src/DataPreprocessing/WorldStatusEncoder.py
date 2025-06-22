import itertools
from collections import defaultdict

import numpy as np
from src.DataPreprocessing.ObjectEncoder import ObjectEncoder
from src.DataPreprocessing.ChangeDetector import ChangeDetector


class WorldStatusEncoder:
    def __init__(self, object_encoder: ObjectEncoder):
        self.object_encoder = object_encoder

    def encode(self, objects: list[dict]) -> np.array:
        object_encodings = []

        for obj in objects:
            obj_encoding = self.object_encoder.encode(obj)
            object_encodings.append(obj_encoding)

        return np.array(object_encodings)

    def encode_matching_order(self, original_objects: list[dict], objects_to_encode: list[dict]):
        """Encodes `objects_to_encode` and makes sure there is a positional matching between `objects_to_encode`
        and `original_objects`, using empty_encodings where no match is found"""
        ordered_objects_to_encode: list[dict | None] = []

        for obj in original_objects:
            obj_id = obj["objectId"]
            corresponding_object: dict | None = ChangeDetector.get_object_by_id(objects_to_encode, obj_id)
            ordered_objects_to_encode.append(corresponding_object)

        return self.encode(ordered_objects_to_encode)

    def decode(self, objects: np.array) -> list[dict]:
        splitted_object_embeddings = np.array_split(objects, self.object_encoder.object_encoding_size)
        decoded_objects = []
        for embedding in splitted_object_embeddings:
            decoded_object = self.object_encoder.decode(embedding)
            decoded_objects.append(decoded_object)

        return decoded_objects


class MostChangesWorldStatusEncoder(WorldStatusEncoder):
    def __init__(self, object_encoder: ObjectEncoder, number_of_significant_objects: int = 3):
        super().__init__(object_encoder)
        self.number_of_significant_objects = number_of_significant_objects
        self.change_detector = ChangeDetector()

    def _get_most_changed_objects(self, objects: dict) -> list[dict]:
        """
        Returns the objects with the most number of changes.
        The number of items that will be returned is self.number_of_significant_objects.
        If there are not enough items, None will be used for padding.
        """
        action_effect = self.change_detector.find_changes_in_file(objects)
        objects_and_their_number_of_changes = {obj: obj.number_of_changes() for obj in action_effect.object_changes}
        del action_effect

        # Sort by most changes
        sorted_objects = sorted(objects_and_their_number_of_changes.items(), key=lambda x: x[1], reverse=True)
        del objects_and_their_number_of_changes

        # Build and return the output list
        return [sorted_objects[i]
                if i < len(sorted_objects)
                else None  # pad with None
                for i in range(0, self.number_of_significant_objects)]

    def encode(self, objects: dict) -> np.array:
        objects = self._get_most_changed_objects(objects)
        return super().encode(objects)

    def decode(self, objects: np.array) -> list[dict]:
        return super().decode(objects)
