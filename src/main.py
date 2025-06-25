import numpy as np
from tqdm import tqdm

from ObjectStore.MetadataObjectStore import MetadataObjectStore
from DataPreprocessing.ChangeDetector import ChangeDetector
from DataAnalysis.DatasetAnalysis import DatasetAnalyzer
from DataPreprocessing.WorldStatusEncoder import MostChangesWorldStatusEncoder
from DataPreprocessing.ObjectEncoder import ObjectEncoder
from src.Predictors.ClassifierManager import ClassifierManager
from src.Predictors.MyDataset import MyDataset

NUMBER_OF_ACTION_SIGNIFICANT_OBJECTS = 10
object_store = MetadataObjectStore("../../ai2thor-hugo/objects/")
whole_dataset: MyDataset = MyDataset(object_store,
      use_cache=True,
      cache_location="../data/dataset_cache/",
      number_of_significant_objects=NUMBER_OF_ACTION_SIGNIFICANT_OBJECTS)
classifier_manager = ClassifierManager(
    object_store=object_store,
    dataset=whole_dataset,
    number_of_significant_objects=NUMBER_OF_ACTION_SIGNIFICANT_OBJECTS,
    model_save_path = "predictor.pth",
    # device="cuda"
)
#classifier_manager.train()

object_encoder = ObjectEncoder()
world_status_encoder = MostChangesWorldStatusEncoder(object_encoder, number_of_significant_objects=NUMBER_OF_ACTION_SIGNIFICANT_OBJECTS)

for file_path in tqdm(object_store.list_files()):
    action_data = object_store.load(file_path) # load the file from disk

    # Prepare the world encoding
    world_encoding = world_status_encoder.encode_action_data(action_data)
    world_encoding = np.array(world_encoding, dtype=np.float32)

    # Prepare network input data
    network_input_shape = (1, world_status_encoder.total_objects_in_world_encoding, world_status_encoder.object_encoder.object_encoding_size) # batch size, how many objects per network input, object encoding length
    world_encoding = np.reshape(world_encoding, network_input_shape)

    # Run inference
    action_class, action_class_name, object_index = classifier_manager.inference(world_encoding)

    # Retrieve the action target encoding from the world encoding and the object index inferred by the network
    action_target_encoding = world_encoding[0, object_index, :] # batch, object index in the world encoding, the entirety of the object encoding
    decoded_object = world_status_encoder.object_encoder.decode(list(action_target_encoding)) # decode the object

    true_action_name = action_data["action_name"]
    true_object_id = action_data["action_objective_id"]
    decoded_object_id = decoded_object["objectId"]

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

