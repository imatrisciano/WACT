from src.Models.ActionEffect import ActionEffect
from src.Models.ObjectChange import ObjectChange
from src.Models.PropertyChange import PropertyChange


class ChangeDetector:
    def __init__(self):
        self.float_tolerance: float = 1e-2
        self.rotation_tolerance = 3 # degrees
        self.properties_to_be_ignored: list[str] = [
            "axisAlignedBoundingBox",
            "objectOrientedBoundingBox",
            "distance"
        ]

    def find_changes_in_file(self, action_data: dict) -> ActionEffect:
        return self.find_changes(action_data["action_name"],
                                 action_data["action_objective_id"],
                                 action_data["before_world_status"],
                                 action_data["after_world_status"],
                                 action_data["liquid"])

    def find_changes(self,
                     action_name: str,
                     action_objective: str,
                     before: dict,
                     after: dict,
                     liquid: str = None) -> ActionEffect:
        changes: list[ObjectChange] = []

        for old_object in before["objects"]:
            new_object: dict | None = self.get_object_by_id(after["objects"], old_object["objectId"])
            object_changes = self.find_changes_in_object(old_object, new_object)
            changes.append(object_changes)

        return ActionEffect(action_name, action_objective, before, changes, liquid)

    def find_changes_in_object(self, before_object, after_object) -> ObjectChange | None:
        """
        Returns a ObjectChange containing changed properties for this object.
        :param before_object: object description before the action
        :param after_object: object description after the action
        :return: a ObjectChange
        """

        if after_object is None:
            # object disappeared
            return ObjectChange(before_object["object_id"], [], object_disappeared=True)

        changes = self._get_changed_properties(before_object, after_object)
        property_changes: list[PropertyChange] = []

        for change_paths in changes:
            for path in change_paths:
                if not isinstance(path, list):
                    path = [path]  # make it into a list

                # initialize variables to navigate the nested dictionaries
                old_property = before_object
                new_property = after_object

                # navigate the nested dictionaries
                for key in path:
                    old_property = old_property[key]
                    new_property = new_property[key]

                # build a human-readable version of path
                # path_str = ".".join(path)
                # print(f" # '{path_str}' changed '{old_property}' ----> '{new_property}'")
                property_changes.append(PropertyChange(path, old_property, new_property))

        return ObjectChange(before_object["objectId"], property_changes)

    @staticmethod
    def _is_float(string):
        try:
            float(string)
            return True
        except:
            return False

    @staticmethod
    def _is_dictionary(property):
        return isinstance(property, dict)

    @staticmethod
    def _rotation_difference(angle1, angle2):
        # Calculate the difference
        diff = (angle2 - angle1) % 360
        # Adjust to be within -180 to 180 degrees
        if diff > 180:
            diff -= 360
        return diff

    def _did_property_change(self, old_value, new_value, current_path=None) -> bool:
        if new_value == old_value:
            return False

        # this property was changed
        if self._is_float(old_value) and self._is_float(new_value):
            # this property is a number, handle changes considering float tolerances
            if current_path is not None and "rotation" in current_path:
                # we are dealing with a rotation angle, in degrees, we should handle periodicity
                difference = self._rotation_difference(old_value, new_value)
                return difference > self.rotation_tolerance
            else:
                return abs(float(old_value) - float(new_value)) > self.float_tolerance
        else:
            return True

    def _get_changed_properties(self, old, new, path=None) -> list:
        changes = []
        if path is None:
            path = []

        for key, value in old.items():
            if key in self.properties_to_be_ignored:
                # skip those properties, do not ever mark them as changed
                continue

            current_path = list.copy(path)
            current_path.append(key)
            old_value = old[key]
            new_value = new[key]
            if self._is_dictionary(value):
                sub_property_changes = self._get_changed_properties(old_value, new_value, current_path)
                if len(sub_property_changes) > 0:
                    changes.append(sub_property_changes)
            else:
                if self._did_property_change(old_value, new_value, current_path):
                    changes.append(current_path)

        return changes

    @staticmethod
    def get_object_by_id(objects: list[dict], object_id: str) -> dict | None:
        for obj in objects:
            if obj['objectId'] == object_id:
                return obj

        return None
