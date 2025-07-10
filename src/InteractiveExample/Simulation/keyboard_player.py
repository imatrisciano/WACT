# Code based on this open source implementation:
# https://github.com/ByZ0e/AI2Thor_keyboard_player/blob/main/keyboard_player.py

import json
import gzip
import traceback
from typing import Optional

import ai2thor.server
from tqdm import tqdm

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from src.InteractiveExample.AgentHistory.AgentHistoryController import AgentHistoryController
from src.InteractiveExample.Simulation.keyboard_player_constants import perform_constants_fixups
from src.InteractiveExample.Simulation.keyboard_player_utils import *


class KeyboardPlayer:
    def __init__(self, agent_history_controller: Optional[AgentHistoryController] = None):
        perform_constants_fixups()

        self.next_action_is_phantom: bool = False

        self.controller: Controller = None
        self.agent_history_controller: Optional[AgentHistoryController] = agent_history_controller

    def _has_agent_history(self):
        """
        Checks if an agent history controller was provided during initialization
        :return: True if an agent history controller was provided during initialization, False otherwise
        """
        return self.agent_history_controller is not None

    def _keyboard_play(self, top_down_frames, first_view_frames, is_rotate, rotate_per_frame):
        self._show_frames()

        step = 0
        pickup = None
        objectId = None

        while True:
            try:
                keystroke = cv2.waitKey(0)
                step += 1

                if keystroke == ord(actionList["FINISH"]):
                    self._stop_environment()
                    return
                elif keystroke == ord("i"):
                    if self._has_agent_history():
                        self.next_action_is_phantom = not self.next_action_is_phantom
                        print(f"Next action is phantom: {self.next_action_is_phantom}")
                elif keystroke == ord("h"):
                    if self._has_agent_history():
                        self.agent_history_controller.print_history()
                elif keystroke == ord("j"):
                    if self._has_agent_history():
                        self.agent_history_controller.analyze_all_phantom_actions()
                        self.agent_history_controller.print_history()
                else:
                    # Figure out action name and target based on input and user choices
                    action, objectId, pickup = get_action_and_object(keystroke, self.controller, objectId, pickup)
                    if action is None:
                        continue # no valid action was selected, try again

                    # Executes the action and gets the resulting event
                    event = self._execute_action(action, objectId)

                    # Register what happened in the history controller
                    if self._has_agent_history():
                        self._register_action(action, objectId, event)

                    if is_rotate:
                        self._rotate_third_view_camera(rotation_angle=rotate_per_frame)

                    # Show and get frames
                    first_view_frame, top_down_frame = self._show_frames()

                    top_down_frames.append(top_down_frame)
                    first_view_frames.append(first_view_frame)
            except KeyboardInterrupt:
                self._stop_environment()
                return
            except Exception as e:
                print("Error while performing this action:")
                traceback.print_exception(e)



    def _register_action(self, action, objectId, event):
        ignored_actions = ["MoveAhead", "MoveLeft", "MoveRight", "MoveBack", "RotateLeft", "RotateRight", "LookUp", "LookDown"]
        if action in ignored_actions:
            self.agent_history_controller.update_current_world_status(event.metadata)
            return

        action_object = objectId if self._action_has_target(action) else None
        if self.next_action_is_phantom:
            self.agent_history_controller.add_phantom_event(event.metadata)
            self.next_action_is_phantom = False
        else:
            self.agent_history_controller.add_event(action, action_object, event.metadata)

    def _stop_environment(self):
        self.controller.stop()
        cv2.destroyAllWindows()
        print("action: STOP")

    def _rotate_third_view_camera(self, rotation_angle):
        ## rotation third camera
        last_event = self.controller.last_event
        pose = compute_rotate_camera_pose(last_event.metadata["sceneBounds"]["center"],
                                          last_event.metadata["thirdPartyCameras"][0], rotation_angle)
        self.controller.step(
            action="UpdateThirdPartyCamera",
            **pose
        )

    def _execute_action(self, action: str, objectId: Optional[str]) -> ai2thor.server.Event:
        """
        Executes the given action on the given objectId and returns the resulting event
        :param action: Name of the action to execute
        :param objectId: (Optional) The objectId to execute the action on
        :return: Action's resulting event
        """

        if self._action_has_target(action):
            return self.controller.step(action=action, objectId=objectId)
        else:
            return self.controller.step(action=action)

    @staticmethod
    def _action_has_target(action_name: str):
        action_has_target: bool = "Object" in action_name
        return action_has_target

    def _show_frames(self):
        """
        Shows and returns both first view and top view frames from the scene
        :param env: The ai2thor environment
        :return: first_view_frame, top_down_frame
        """

        first_view_frame = self.controller.last_event.frame
        cv2.imshow("first_view", cv2.cvtColor(first_view_frame, cv2.COLOR_RGB2BGR))

        # remove the ceiling
        self.controller.step(action="ToggleMapView")
        top_down_frame = self.controller.last_event.third_party_camera_frames[0]
        cv2.imshow("top_view", cv2.cvtColor(top_down_frame, cv2.COLOR_RGB2BGR))
        self.controller.step(action="ToggleMapView")

        return first_view_frame, top_down_frame


    def run(self, scene_name="FloorPlan205_physics", gridSize=0.25, rotateStepDegrees=15,
             birds_eye_view=False, slope_degree=45, down_angle=65, use_procthor=False, procthor_scene_file="", procthor_scene_num=100,
             is_rotate=True, rotate_per_frame=6, generate_video=False, generate_gif=False):

        print("You can play using the following keys:")
        for action in actionList:
            key = actionList[action]
            print(f" [{key.upper()}]: {action}")
        print()


        ## procthor room
        if use_procthor:
            with gzip.open(procthor_scene_file, "r") as f:
                houses = [line for line in tqdm(f, total=10000, desc=f"Loading train")]
            ## procthor train set's room
            house = json.loads(houses[procthor_scene_num])
        else:
            ## select room, 1-30，201-230，301-330，401-430 are ithor's room
            house = scene_name

        self.controller = Controller(
            agentMode="default",
            visibilityDistance=5,
            renderInstanceSegmentation=True,
            scene=house,
            # step sizes
            gridSize=gridSize,
            snapToGrid=False,
            rotateStepDegrees=rotateStepDegrees,
            # camera properties
            width=1200,
            height=800,
            fieldOfView=90,
            platform=CloudRendering,
        )

        ## add third view camera
        event = self.controller.step(action="GetMapViewCameraProperties")
        ## third camera's fov
        third_fov = 60

        if not birds_eye_view:
            ## top_view(slope)
            pose = initialize_side_camera_pose(event.metadata["sceneBounds"], event.metadata["actionReturn"], third_fov,
                                               slope_degree, down_angle)
        else:
            ## BEV
            pose = event.metadata["actionReturn"]
            is_rotate = False  ## assume that BEV do not need rotation

        event = self.controller.step(
            action="AddThirdPartyCamera",
            skyboxColor="black",
            fieldOfView=third_fov,
            **pose
        )

        # Bootstrap the agent history controller
        if self._has_agent_history():
            self.agent_history_controller.set_initial_world_status(event.metadata)

        ## collect frame
        first_view_frames = []
        third_view_frames = []

        try:
            ## use keyboard control agent
            self._keyboard_play(third_view_frames, first_view_frames, is_rotate, rotate_per_frame)
        except KeyboardInterrupt:
            print("\nSession interrupted by the user.\n")
            self._stop_environment()

        if generate_video:
            save_videos(scene_name, first_view_frames, third_view_frames)

        if generate_gif:
            save_gifs(scene_name, first_view_frames, third_view_frames)


if __name__ == "__main__":
    player = KeyboardPlayer()
    player.run(scene_name="FloorPlan19_physics",  ## room
         gridSize=0.25, rotateStepDegrees=15,  ## agent step len and rotate degree
         birds_eye_view=False,  ## Bird's-eye view or top view(slope)
         slope_degree=60,  ## top view(slope)'s initial rotate degree
         down_angle=65,  ## top view(slope)'s pitch angle, should be 0-90, 90 equal to Bird's-eye view
         use_procthor=False,  ## use procthor room, True: select room from procthor train set, need dataset dir
         procthor_scene_file="",  ## procthor train set dir
         procthor_scene_num=100,  ## select scene from procthor train set
         is_rotate=True,  ## top_view rotate?
         rotate_per_frame=6,  ## top_view rotate degree
         generate_video=True,  ## use frames generate video
         generate_gif=True,  ## use frames generate gif
         )