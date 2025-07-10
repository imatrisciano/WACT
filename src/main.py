from tqdm import tqdm
from src.Predictors.PredictorPipeline import PredictorPipeline


predictor_pipeline = PredictorPipeline()

for file_path in tqdm(predictor_pipeline.list_training_files()):
    detected_action_name, detected_object_id, true_action_name, true_object_id = predictor_pipeline.predict_from_file(file_path)

    print(f" [Truth]:      Action: {true_action_name} on object {true_object_id}")
    print(f" [Prediction]: Action: {detected_action_name} on object {detected_object_id}")
    print("")

exit(0)
