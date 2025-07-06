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

perform_constants_fixups()


VIDEO_PATH = "../../data/video/"
GIF_PATH = "../../data/gif/"


def keyboard_play(env, top_down_frames, first_view_frames, is_rotate, rotate_per_frame):
    first_view_frame = env.last_event.frame
    cv2.imshow("first_view", cv2.cvtColor(first_view_frame, cv2.COLOR_RGB2BGR))

    # remove the ceiling
    env.step(action="ToggleMapView")
    top_down_frame = env.last_event.third_party_camera_frames[0]
    cv2.imshow("top_view", cv2.cvtColor(top_down_frame, cv2.COLOR_RGB2BGR))
    env.step(action="ToggleMapView")

    step = 0
    pickup = None
    objectId = None

    while True:
        keystroke = cv2.waitKey(0)
        step += 1

        if keystroke == ord(actionList["FINISH"]):
            env.stop()
            cv2.destroyAllWindows()
            print("action: STOP")
            break

        action, objectId, pickup = get_action_and_object(keystroke, env, objectId, pickup)
        if action is None:
            continue

        # agent step
        action_has_target: bool = "Object" in action
        if action_has_target:
            env.step(action=action, objectId=objectId)
        else:
            env.step(action=action)

        if is_rotate:
            ## rotation third camera
            pose = compute_rotate_camera_pose(env.last_event.metadata["sceneBounds"]["center"],
                                              env.last_event.metadata["thirdPartyCameras"][0], rotate_per_frame)

            env.step(
                action="UpdateThirdPartyCamera",
                **pose
            )

        first_view_frame = env.last_event.frame
        cv2.imshow("first_view", cv2.cvtColor(first_view_frame, cv2.COLOR_RGB2BGR))

        # remove the ceiling
        env.step(action="ToggleMapView")
        top_down_frame = env.last_event.third_party_camera_frames[0]
        cv2.imshow("top_view", cv2.cvtColor(top_down_frame, cv2.COLOR_RGB2BGR))
        env.step(action="ToggleMapView")

        top_down_frames.append(top_down_frame)
        first_view_frames.append(first_view_frame)


def main(scene_name="FloorPlan205_physics", gridSize=0.25, rotateStepDegrees=15,
         birds_eye_view=False, slope_degree=45, down_angle=65, use_procthor=False, procthor_scene_file="", procthor_scene_num=100,
         is_rotate=True, rotate_per_frame=6, generate_video=False, generate_gif=False):
    ## procthor room
    if use_procthor:
        with gzip.open(procthor_scene_file, "r") as f:
            houses = [line for line in tqdm(f, total=10000, desc=f"Loading train")]
        ## procthor train set's room
        house = json.loads(houses[procthor_scene_num])
    else:
        ## select room, 1-30，201-230，301-330，401-430 are ithor's room
        house = scene_name

    controller = Controller(
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
    event = controller.step(action="GetMapViewCameraProperties")
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

    event = controller.step(
        action="AddThirdPartyCamera",
        skyboxColor="black",
        fieldOfView=third_fov,
        **pose
    )

    ## collect frame
    first_view_frames = []
    third_view_frames = []

    ## use keyboard control agent
    keyboard_play(controller, third_view_frames, first_view_frames, is_rotate, rotate_per_frame)

    ## use frames generate video
    if generate_video:

        if not os.path.exists(VIDEO_PATH):
            os.mkdir(VIDEO_PATH)

        export_video(VIDEO_PATH + "first_view_{}.mp4".format(scene_name), first_view_frames)
        export_video(VIDEO_PATH + "third_view_{}.mp4".format(scene_name), third_view_frames)

    ## use frames generate gif
    if generate_gif:

        if not os.path.exists(VIDEO_PATH):
            os.mkdir(VIDEO_PATH)

        clip = show_video(third_view_frames, fps=5)
        clip.write_gif(VIDEO_PATH + "third_view_{}.gif".format(scene_name))
        clip2 = show_video(first_view_frames, fps=5)
        clip2.write_gif(VIDEO_PATH + "first_view_{}.gif".format(scene_name))


if __name__ == "__main__":
    main(scene_name="FloorPlan19_physics",  ## room
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