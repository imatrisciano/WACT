import numpy as np
from tqdm import tqdm

from ObjectStore.MetadataObjectStore import MetadataObjectStore
from DataPreprocessing.ChangeDetector import ChangeDetector
from DataAnalysis.DatasetAnalysis import DatasetAnalyzer
from DataPreprocessing.WorldStatusEncoder import MostChangesWorldStatusEncoder
from DataPreprocessing.ObjectEncoder import ObjectEncoder
from src.Predictors.ClassifierManager import ClassifierManager
from src.Predictors.MyDataset import MyDataset

object_store = MetadataObjectStore("../../ai2thor-hugo/objects/")
classifier_manager = ClassifierManager(object_store)
#classifier_manager.train()

object_encoder = ObjectEncoder()
world_status_encoder = MostChangesWorldStatusEncoder(object_encoder, number_of_significant_objects=10)

for file_path in tqdm(object_store.list_files()):
    obj = object_store.load(file_path)
    world_encoding = world_status_encoder.encode_action_data(obj)
    world_encoding = np.array(world_encoding, dtype=np.float32)
    world_encoding = np.reshape(world_encoding, (1,20,37))

    action_class, action_class_name, object_index = classifier_manager.inference(world_encoding)

    decoded_object = world_status_encoder.object_encoder.decode(list(world_encoding[0, object_index, :]))
    decoded_object_id = decoded_object["objectId"]

    true_action_name = obj["action_name"]
    true_object_id = obj["action_objective_id"]

    dataset = classifier_manager.whole_dataset

    print(f" [Truth]:      Action: {dataset.label2id(true_action_name)} ({true_action_name}) on object - ({true_object_id})")
    print(f" [Prediction]: Action: {action_class} ({action_class_name}) on object {object_index} ({decoded_object_id})")
    print("")

exit(0)


#object_store = MetadataObjectStore("../../ai2thor-hugo/objects/")
#change_detector = ChangeDetector()
#
#object_encoder = ObjectEncoder()
#world_status_encoder = MostChangesWorldStatusEncoder(object_encoder, number_of_significant_objects=10)
#
#dataset_analyzer = DatasetAnalyzer(object_store, change_detector)
#dataset_analyzer.plot_dataset_info()
#
#
#for file_path in tqdm(object_store.list_files()):
#    obj = object_store.load(file_path)
#    world_encoding = world_status_encoder.encode_action_data(obj)
#
#    print(world_encoding)

