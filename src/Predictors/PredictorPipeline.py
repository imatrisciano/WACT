import os

import numpy as np

from src.DataPreprocessing.ObjectEncoder import ObjectEncoder
from src.DataPreprocessing.WorldStatusEncoder import MostChangesWorldStatusEncoder
from src.ObjectStore.MetadataObjectStore import MetadataObjectStore
from src.Predictors.ClassifierManager import ClassifierManager
from src.Predictors.MyDataset import MyDataset


class PredictorPipeline:
    """
    Implements an end to end predictor pipeline for Which Action Caused This
    """

    def __init__(self):
        NUMBER_OF_ACTION_SIGNIFICANT_OBJECTS = 10

        self.object_store = MetadataObjectStore("../../ai2thor-hugo/objects/")
        self.whole_dataset: MyDataset = MyDataset(self.object_store,
                                             use_cache=True,
                                             cache_location="../data/dataset_cache/",
                                             number_of_significant_objects=NUMBER_OF_ACTION_SIGNIFICANT_OBJECTS)
        self.classifier_manager = ClassifierManager(
            object_store=self.object_store,
            dataset=self.whole_dataset,
            number_of_significant_objects=NUMBER_OF_ACTION_SIGNIFICANT_OBJECTS,
            model_save_path="predictor.pth",
            # device="cuda"
        )
        # classifier_manager.train()

        object_encoder = ObjectEncoder()
        self.world_status_encoder = MostChangesWorldStatusEncoder(object_encoder,
                                                             number_of_significant_objects=NUMBER_OF_ACTION_SIGNIFICANT_OBJECTS)

    def train(self):
        self.classifier_manager.train()

    def predict(self, action_data: dict) -> (str, str):
        """
        Performs action and action target prediction given the action data
        :param action_data: A dictionary containing 'before_world_status', 'after_world_status', 'action_name', 'action_objective_id' and 'liquid' keys
        :return: The action class name and the action target object id
        """

        # Prepare the world encoding
        world_encoding = self.world_status_encoder.encode_action_data(action_data)
        world_encoding = np.array(world_encoding, dtype=np.float32)

        # Prepare network input data
        network_input_shape = (1, self.world_status_encoder.total_objects_in_world_encoding,
                               self.world_status_encoder.object_encoder.object_encoding_size)  # batch size, how many objects per network input, object encoding length
        world_encoding = np.reshape(world_encoding, network_input_shape)

        # Run inference
        action_class, action_class_name, object_index, predicted_action_confidence, predicted_object_confidence = self.classifier_manager.inference(world_encoding)

        # Retrieve the action target encoding from the world encoding and the object index inferred by the network
        action_target_encoding = world_encoding[0, object_index, :]  # batch, object index in the world encoding, the entirety of the object encoding
        decoded_object = self.world_status_encoder.object_encoder.decode(action_target_encoding)  # decode the object

        decoded_object_id = decoded_object["objectId"]
        return action_class_name, decoded_object_id, predicted_action_confidence, predicted_object_confidence

    def predict_from_file(self, file_path):
        """
        Loads an action file from disk and performs prediction on it
        :param file_path: The action data file
        :return: detected_action_name: str, detected_object_id: str, true_action_name: str, true_object_id: str
        """

        action_data = self.object_store.load(file_path)  # load the file from disk

        detected_action_name, detected_object_id, predicted_action_confidence, predicted_object_confidence = self.predict(action_data)

        true_action_name = action_data["action_name"]
        true_object_id = action_data["action_objective_id"]

        return detected_action_name, detected_object_id, predicted_action_confidence, predicted_object_confidence, true_action_name, true_object_id

    def predict_from_before_and_after_object_lists(self, objects_before_action: list, objects_after_action: list):
        action_data = {
            "before_world_status": {
                "objects": objects_before_action
            },
            "after_world_status": {
                "objects": objects_after_action
            },
            "action_name": "",
            "action_objective_id": "",
            "liquid": ""
        }
        return self.predict(action_data)

    def list_training_files(self) -> list[os.PathLike | str]:
        return self.object_store.list_files()