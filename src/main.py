from tqdm import tqdm

from src.InteractiveExample.AgentHistory.AgentHistoryController import AgentHistoryController
from src.InteractiveExample.Simulation.keyboard_player import KeyboardPlayer
from src.Predictors.PredictorPipeline import PredictorPipeline


predictor_pipeline = PredictorPipeline()

agent_history_controller = AgentHistoryController(predictor_pipeline)
player = KeyboardPlayer(agent_history_controller)
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


for file_path in tqdm(predictor_pipeline.list_training_files()):
    detected_action_name, detected_object_id, predicted_action_confidence, predicted_object_confidence, true_action_name, true_object_id = predictor_pipeline.predict_from_file(file_path)

    print(f" [Truth]:      Action: {true_action_name} on object {true_object_id}")
    print(f" [Prediction]: Action: {detected_action_name} on object {detected_object_id}")
    print("")

exit(0)
