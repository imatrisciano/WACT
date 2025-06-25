from tqdm import tqdm

from ObjectStore.MetadataObjectStore import MetadataObjectStore
from DataPreprocessing.ChangeDetector import ChangeDetector
from DataAnalysis.DatasetAnalysis import DatasetAnalyzer
from DataPreprocessing.WorldStatusEncoder import MostChangesWorldStatusEncoder
from DataPreprocessing.ObjectEncoder import ObjectEncoder
from src.Predictors.ClassifierManager import ClassifierManager

object_store = MetadataObjectStore("../../ai2thor-hugo/objects/")
classifier_manager = ClassifierManager(object_store)
classifier_manager.train()
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

