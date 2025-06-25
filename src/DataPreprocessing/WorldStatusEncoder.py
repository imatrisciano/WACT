from src.DataPreprocessing.ObjectEncoder import ObjectEncoder
from src.DataPreprocessing.ChangeDetector import ChangeDetector


class WorldStatusEncoder:
    def __init__(self, object_encoder: ObjectEncoder):
        self.object_encoder = object_encoder

    def encode(self, objects: list[dict]) -> list[float]:
        object_encodings = []

        for obj in objects:
            obj_encoding = self.object_encoder.encode(obj)
            object_encodings.append(obj_encoding)

        return object_encodings

    def encode_matching_order(self, original_objects: list[dict], objects_to_encode: list[dict]):
        """Encodes `objects_to_encode` and makes sure there is a positional matching between `objects_to_encode`
        and `original_objects`, using empty_encodings where no match is found"""
        ordered_objects_to_encode: list[dict | None] = []

        for obj in original_objects:
            obj_id = obj["objectId"]
            corresponding_object: dict | None = ChangeDetector.get_object_by_id(objects_to_encode, obj_id)
            ordered_objects_to_encode.append(corresponding_object)

        return self.encode(ordered_objects_to_encode)

    def encode_before_and_after_world_status(self, before_objects: list[dict], after_objects: list[dict]) -> list[float]:
        before_objects_encodings = self.encode(before_objects)
        after_objects_encodings = self.encode_matching_order(before_objects, after_objects)

        # concatenate lists and return them
        return before_objects_encodings + after_objects_encodings

    def encode_action_data(self, action_data: dict) -> list[float]:
        return self.encode_before_and_after_world_status(action_data["before_world_status"], action_data["after_world_status"])

    def decode(self, objects: list) -> list[dict]:
        decoded_objects = []

        for i in range(0, len(objects), self.object_encoder.object_encoding_size):
            current_object_encoding = objects[i:i + self.object_encoder.object_encoding_size]
            decoded_object = self.object_encoder.decode(current_object_encoding)
            decoded_objects.append(decoded_object)

        return decoded_objects


class MostChangesWorldStatusEncoder(WorldStatusEncoder):
    def __init__(self, object_encoder: ObjectEncoder, number_of_significant_objects: int = 3):
        super().__init__(object_encoder)
        self.number_of_significant_objects = number_of_significant_objects
        self.change_detector = ChangeDetector()

    def _get_most_changed_objects(self, action_data: dict) -> list[dict]:
        """
        Returns the objects with the most number of changes.
        The number of items that will be returned is self.number_of_significant_objects.
        If there are not enough items, None will be used for padding.
        """
        action_effect = self.change_detector.find_changes_in_file(action_data)
        objects_and_their_number_of_changes = {obj: obj.get_change_score() for obj in action_effect.object_changes}
        del action_effect

        # Sort by most changes
        sorted_objects = sorted(objects_and_their_number_of_changes.items(), key=lambda x: x[1], reverse=True)
        del objects_and_their_number_of_changes

        get_object_by_its_object_change = lambda object_change: (
            ChangeDetector.get_object_by_id(action_data["before_world_status"]["objects"], object_change.object_id))


        # Build and return the output list
        return [get_object_by_its_object_change(sorted_objects[i][0])
                if i < len(sorted_objects)
                else None  # pad with None
                for i in range(0, self.number_of_significant_objects)]

    def encode_action_data(self, action_data: dict) -> list[float]:
        sorted_before_objects = self._get_most_changed_objects(action_data)
        after_objects = action_data["after_world_status"]["objects"]
        return super().encode_before_and_after_world_status(sorted_before_objects, after_objects)
