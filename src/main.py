from tqdm import tqdm

print("Importing modules...")
from src.DataAnalysis.DatasetAnalysis import DatasetAnalyzer
from src.DataPreprocessing.ChangeDetector import ChangeDetector
from src.InteractiveExample.AgentHistory.AgentHistoryController import AgentHistoryController
from src.InteractiveExample.LanguageModel.wact_chatbot import WACTChatBot
from src.InteractiveExample.Simulation.keyboard_player import KeyboardPlayer
from src.Predictors.PredictorPipeline import PredictorPipeline


predictor_pipeline = PredictorPipeline()

def analyze_dataset():
    dataset_analyzer = DatasetAnalyzer(object_store=predictor_pipeline.object_store, change_detector=ChangeDetector())
    dataset_analyzer.plot_dataset_info()

def play():
    agent_history_controller = AgentHistoryController(predictor_pipeline)
    chatbot = WACTChatBot(agent_history_controller, model_name="qwen3:8b")

    player = KeyboardPlayer(agent_history_controller, chatbot)
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

def launch_classifier():
    try:
        for file_path in tqdm(predictor_pipeline.list_training_files()):
            detected_action_name, detected_object_id, predicted_action_confidence, predicted_object_confidence, true_action_name, true_object_id = predictor_pipeline.predict_from_file(
                file_path)

            print(f" [Truth]:      Action: {true_action_name} on object {true_object_id}")
            print(f" [Prediction]: Action: {detected_action_name} on object {detected_object_id}")
            print("")
    except KeyboardInterrupt:
        print("Stopped by the user")

def choose_option():
    """
    Lets the user pick an option to use this tool
    """

    # All the available options, expressed as "option text": option_function
    options = {
        "Analyze dataset": analyze_dataset,
        "Dataset grid search (to find the best parameters)": predictor_pipeline.grid_search,
        "Train classifier (with known best parameters)": predictor_pipeline.train,
        "Launch classifier": launch_classifier,
        "Play": play,
        "Exit": exit
    }

    # Present the options
    print("What are you up to?")
    for index, option_name in enumerate(options.keys()):
        print(f"[{index}]: {option_name}")
    print()

    # Keep on asking the next thing to do until the user decides to exit
    while True:
        user_choice = input("> ")
        try:
            # Convert user text to int, will throw a ValueError if it fails
            user_choice = int(user_choice.strip())

            # Boundaries check
            if user_choice < 0 or user_choice >= len(options):
                raise ValueError

            # Checks passed, invoke that function
            chosen_entry = list(options.keys())[user_choice]
            function_to_call = options[chosen_entry]
            function_to_call() # call it
            break
        except ValueError:
            print("Invalid choice")
        except KeyboardInterrupt:
            print("Goodbye")
            exit(0)

if __name__ == "__main__":
    while True:
        choose_option()
        print()
        print()

