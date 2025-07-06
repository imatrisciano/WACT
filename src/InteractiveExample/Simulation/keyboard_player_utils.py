import math
from typing import Sequence

import cv2
import numpy as np
from moviepy import ImageSequenceClip

from src.InteractiveExample.Simulation.keyboard_player_constants import VAL_ACTION_OBJECTS, VAL_RECEPTACLE_OBJECTS, \
    actionList


def show_video(frames: Sequence[np.ndarray], fps: int = 10):
    """Show a video composed of a sequence of frames.

    Example:
    frames = [
        controller.step("RotateRight", degrees=5).frame
        for _ in range(72)
    ]
    show_video(frames, fps=5)
    """
    frames = ImageSequenceClip(frames, fps=fps)
    return frames


def export_video(path, frames):
    """Merges all the saved frames into a .mp4 video and saves it to `path`"""

    video = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        5,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    for frame in frames:
        # assumes that the frames are RGB images. CV2 uses BGR.
        video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.destroyAllWindows()
    video.release()


def compute_rotate_camera_pose(center, pose, degree_per_frame=6):
    """degree_per_frame: set the degree of rotation for each frame"""

    def rotate_pos(x1, y1, x2, y2, degree):
        angle = math.radians(degree)
        n_x1 = (x1 - x2) * math.cos(angle) - (y1 - y2) * math.sin(angle) + x2
        n_y1 = (x1 - x2) * math.sin(angle) + (y1 - y2) * math.cos(angle) + y2

        return n_x1, n_y1

    # print(math.sqrt((pose["position"]["x"]-center["x"])**2 + (pose["position"]["z"]-center["z"])**2))
    x, z = rotate_pos(pose["position"]["x"], pose["position"]["z"], center["x"], center["z"], degree_per_frame)
    pose["position"]["x"], pose["position"]["z"] = x, z

    direction_x = center["x"] - x
    direction_z = center["z"] - z
    pose["rotation"]["y"] = math.degrees(math.atan2(direction_x, direction_z))

    # print(math.sqrt((pose["position"]["x"]-center["x"])**2 + (pose["position"]["z"]-center["z"])**2))

    return pose


def initialize_side_camera_pose(scene_bound, pose, third_fov=60, slope_degree=45, down_angle=70, scale_factor=8):
    """
    down_angle: the x-axis rotation angle of the camera, represents the top view of the front view from top to bottom, which needs to be less than 90 degrees
    ensure the line vector between scene's center & camera 's angel equal down_angle
    scale_factor scale the camera's view, make it larger ensure camera can see the whole scene
    """
    fov_rad = np.radians(third_fov)
    pitch_rad = np.radians(down_angle)
    distance = (scene_bound["center"]["y"] / 2) / np.tan(fov_rad / 2)
    pose["position"]["y"] = scene_bound["center"]["y"] + distance * scale_factor * np.sin(pitch_rad)
    pose["position"]["z"] = scene_bound["center"]["z"] - distance * scale_factor * np.cos(pitch_rad)

    pose["rotation"]["x"] = down_angle

    pose["orthographic"] = False
    del pose["orthographicSize"]

    pose = compute_rotate_camera_pose(scene_bound["center"], pose, slope_degree)

    return pose

def get_action_and_object(keystroke, env, objectId, pickup) -> tuple:
    if keystroke == ord(actionList["MoveAhead"]):
        action = "MoveAhead"
        print("action: MoveAhead")
    elif keystroke == ord(actionList["MoveBack"]):
        action = "MoveBack"
        print("action: MoveBack")
    elif keystroke == ord(actionList["MoveLeft"]):
        action = "MoveLeft"
        print("action: MoveLeft")
    elif keystroke == ord(actionList["MoveRight"]):
        action = "MoveRight"
        print("action: MoveRight")
    elif keystroke == ord(actionList["RotateLeft"]):
        action = "RotateLeft"
        print("action: RotateLeft")
    elif keystroke == ord(actionList["RotateRight"]):
        action = "RotateRight"
        print("action: RotateRight")
    elif keystroke == ord(actionList["LookUp"]):
        action = "LookUp"
        print("action: LookUp")
    elif keystroke == ord(actionList["LookDown"]):
        action = "LookDown"
        print("action: LookDown")

    elif keystroke == ord(actionList["PickupObject"]):
        action = "PickupObject"
        objectId = get_interact_object(env, action)
        pickup = objectId.split('|')[0]
        print("action: PickupObject")
    elif keystroke == ord(actionList["PutObject"]):
        action = "PutObject"
        print('holding', pickup)
        objectId = get_interact_object(env, action, pickup=pickup)
        print("action: PutObject")
    elif keystroke == ord(actionList["OpenObject"]):
        action = "OpenObject"
        objectId = get_interact_object(env, action)
        print("action: OpenObject")
    elif keystroke == ord(actionList["CloseObject"]):
        action = "CloseObject"
        objectId = get_interact_object(env, action)
        print("action: CloseObject")
    elif keystroke == ord(actionList["ToggleObjectOn"]):
        action = "ToggleObjectOn"
        objectId = get_interact_object(env, action)
        print("action: ToggleObjectOn")
    elif keystroke == ord(actionList["ToggleObjectOff"]):
        action = "ToggleObjectOff"
        objectId = get_interact_object(env, action)
        print("action: ToggleObjectOff")
    elif keystroke == ord(actionList["SliceObject"]):
        action = "SliceObject"
        objectId = get_interact_object(env, action)
        print("action: SliceObject")
    else:
        print("INVALID KEY", keystroke)
        return None, None

    return action, objectId, pickup

def get_interact_object(env, action, pickup=None):
    candidates = []
    objectId = ''
    interactable_obj_list = []
    if action == 'PickupObject':
        interactable_obj_list = VAL_ACTION_OBJECTS['Pickupable']
    elif action == 'PutObject':
        for recep, objs in VAL_RECEPTACLE_OBJECTS.items():
            if pickup in objs:
                interactable_obj_list.append(recep)
    elif action == 'OpenObject' or action == 'CloseObject':
        interactable_obj_list = VAL_ACTION_OBJECTS['Openable']
    elif action == 'ToggleObjectOn' or action == 'ToggleObjectOff':
        interactable_obj_list = VAL_ACTION_OBJECTS['Toggleable']
    elif action == 'SliceObject':
        interactable_obj_list = VAL_ACTION_OBJECTS['Sliceable']

    for obj in env.last_event.metadata["objects"]:
        if obj["objectId"] in env.last_event.instance_masks.keys() and obj["visible"] and obj["objectId"].split('|')[
            0] in interactable_obj_list:
            if obj["objectId"].startswith('Sink') and not obj["objectId"].endswith('SinkBasin'):
                print(obj["objectId"])
                continue
            if obj["objectId"].startswith('Bathtub') and not obj["objectId"].endswith('BathtubBasin'):
                continue
            candidates.append(obj["objectId"])
    if len(candidates) == 0:
        print('no valid interact object candidates')
        return None
    else:
        print('===========choose index from the candidates==========')
        for index, obj in enumerate(candidates):
            print(index, ':', obj)
        while True:
            # input the index of candidates in the console
            keystroke = input()
            print(keystroke)
            if keystroke == actionList["FINISH"]:
                print("stop interact")
                break
            try:
                objectId = candidates[int(keystroke)]
            except:
                print("INVALID KEY", keystroke)
                continue
            print(objectId)
            break
        return objectId