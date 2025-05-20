from src.Models.ObjectChange import ObjectChange


class ActionEffect:
    def __init__(self, action_name: str, action_objective: str, before_status: dict, object_changes: list[ObjectChange],
                 liquid: str = None):
        self.action_name: str = action_name
        self.action_objective: str = action_objective
        self.liquid: str = liquid
        self.before_status: dict = before_status
        self.object_changes: list[ObjectChange] = object_changes

    def __str__(self):
        """
        Returns a human-readable summary of the action effect.
        Only includes changed objects.
        """
        output: str = f"Action '{self.action_name}' performed on object '{self.action_objective}' and (optionally) on liquid '{self.liquid}': \n"
        for changed_object in [x for x in self.object_changes if x.has_changes()]:
            output += f"\tObject '{changed_object.object_id}' changed:\n"
            for changed_property in changed_object.object_changes:
                output += f"\t\t Property {changed_property.property_path} changed from '{changed_property.before_value}' to '{changed_property.after_value}'\n"
                pass
            if changed_object.object_disappeared:
                output += "\t\tObject disappeared\n"

        return output
