from typing import Optional


class AgentActionInfo:
    def __init__(self, action: Optional[str], action_target: Optional[str], after_event_world_status):
        self.action: Optional[str] = action
        self.action_target: Optional[str] = action_target
        self.after_event_world_status = after_event_world_status


        self.action_confidence = 0.0 if action is None else 1.0
        self.target_confidence = 0.0 if action_target is None else 1.0

        # If the action info was phantom on creation
        self.was_phantom_action = self.is_phantom()

    def is_phantom(self):
        """
        Returns whether this action is a phantom action
        :return: True if the action is a phantom action, False otherwise
        """
        return self.action is None

    def was_phantom(self):
        """
        Returns whether this action was a phantom action upon creation
        :return: True if the action was a phantom action, False otherwise
        """
        return self.was_phantom_action

    def has_target_object(self):
        """
        Returns whether this action has a target object
        :return: True if the action has a target object, False otherwise
        """
        return self.action_target is not None

    def get_short_description(self) -> str:
        """
        Gets a short string describing this action
        :return: A short action description
        """

        if self.is_phantom():
            return "<Unanalyzed phantom event>"

        action_description = f"{self.action} ({(self.action_confidence * 100.0):.2f}%)"
        if self.has_target_object():
            action_description += f" on '{self.action_target}' ({(self.target_confidence * 100.0):.2f}%)"

        if self.was_phantom():
            action_description = f"[PHANTOM] {action_description}"

        return action_description