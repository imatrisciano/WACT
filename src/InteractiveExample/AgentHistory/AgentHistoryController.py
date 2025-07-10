from sys import stderr
from typing import Optional

from src.InteractiveExample.AgentHistory.AgentActionInfo import AgentActionInfo
from src.Predictors.PredictorPipeline import PredictorPipeline


class AgentHistoryController:
    def __init__(self, predictor_pipeline: PredictorPipeline):
        self.initial_world_status = None
        self.agent_action_history: list[AgentActionInfo] = []
        self.predictor_pipeline = predictor_pipeline

    def add_event(self, action: str, action_target: Optional[str], after_event_world_status):
        """
        Adds an event to the event history
        :param action: The name of the performed action
        :param action_target: (Optional) The objectId of the target of the action
        :param after_event_world_status: The world status after the action was performed
        """

        action_info = AgentActionInfo(action, action_target, after_event_world_status)
        self.agent_action_history.append(action_info)

    def add_phantom_event(self, after_event_world_status):
        """
        Adds a phantom event to the event history
        :param after_event_world_status: The world status after a phantom action was observed
        """

        action_info = AgentActionInfo(None, None, after_event_world_status)
        self.agent_action_history.append(action_info)

    def set_initial_world_status(self, initial_world_status):
        self.initial_world_status = initial_world_status

    def print_history(self):
        if len(self.agent_action_history) == 0:
            print("The agent execution history is empty")
            return

        print("Here's the agent's execution history:")
        for index, action_info in enumerate(self.agent_action_history):
            print(f" [{index}]: {action_info.get_short_description()}")

        print()

    def get_history(self):
        if len(self.agent_action_history) == 0:
            return "The agent execution history is empty"

        history: str = "Here's the agent's execution history:\n"
        for index, action_info in enumerate(self.agent_action_history):
            history += f" [{index}]: {action_info.get_short_description()}\n"

        return history

    def analyze_all_phantom_actions(self):
        for index, event in enumerate(self.agent_action_history):
            if event.is_phantom():
                self.figure_out_phantom_action(index)

    def update_current_world_status(self, current_world_status):
        if len(self.agent_action_history) == 0:
            self.initial_world_status = current_world_status
        else:
            self.agent_action_history[-1].after_event_world_status = current_world_status

    def figure_out_phantom_action(self, event_index: int) -> (str, str):
        """
        Runs the prediction engine on a given phantom action to figure out which action it was, then updates the action info with the new data
        :param event_index: Index in the event history corresponding to the phantom action
        :return: The detected action name, the detected target object id
        """

        # Boundary checks
        if event_index < 0 or event_index >= len(self.agent_action_history):
            raise IndexError("Event index out of range")

        target_action = self.agent_action_history[event_index]
        if not target_action.is_phantom():
            print("Target event is not a phantom action, prediction skipped", file=stderr)
            return target_action.action, target_action.action_target

        # Figure out before and after event world statuses:
        after_event_world_status = target_action.after_event_world_status['objects']
        if event_index == 0:
            before_event_world_status = self.initial_world_status['objects']
        else:
            before_event_world_status = self.agent_action_history[event_index-1].after_event_world_status['objects']


        # Run inference
        detected_action_name, detected_object_id, predicted_action_confidence, predicted_object_confidence =\
            self.predictor_pipeline.predict_from_before_and_after_object_lists(before_event_world_status, after_event_world_status)

        # Assign detected values to the action
        target_action.action = detected_action_name
        target_action.action_target = detected_object_id

        target_action.action_confidence = predicted_action_confidence
        target_action.target_confidence = predicted_object_confidence

        # Also return the prediction result
        return detected_action_name, detected_object_id


