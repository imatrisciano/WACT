# Code based on this open source implementation:
# https://github.com/ByZ0e/AI2Thor_keyboard_player/blob/main/keyboard_player.py

import json
import gzip
import os
from tqdm import tqdm

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from src.InteractiveExample.Simulation.keyboard_player_constants import perform_constants_fixups
from src.InteractiveExample.Simulation.keyboard_player_utils import *


class KeyboardPlayer:
    def __init__(self):
        perform_constants_fixups()
        self.controller: Controller = None

    def _keyboard_play(self, top_down_frames, first_view_frames, is_rotate, rotate_per_frame):
        self._show_frames()

        step = 0
        pickup = None
        objectId = None

        while True:
            keystroke = cv2.waitKey(0)
            step += 1

            if keystroke == ord(actionList["FINISH"]):
                self._stop_environment()
                return

            action, objectId, pickup = get_action_and_object(keystroke, self.controller, objectId, pickup)
            if action is None:
                continue # no valid action was selected, try again

            self._execute_action(action, objectId)

            if is_rotate:
                self._rotate_third_view_camera(rotation_angle=rotate_per_frame)

            # Show and get frames
            first_view_frame, top_down_frame = self._show_frames()

            top_down_frames.append(top_down_frame)
            first_view_frames.append(first_view_frame)


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

    def _execute_action(self, action, objectId):
        action_has_target: bool = "Object" in action
        if action_has_target:
            self.controller.step(action=action, objectId=objectId)
        else:
            self.controller.step(action=action)

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