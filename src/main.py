from tqdm import tqdm

from ObjectStore.MetadataObjectStore import MetadataObjectStore
from DataPreprocessing.ChangeDetector import ChangeDetector
from DataAnalysis.DatasetAnalysis import DatasetAnalyzer
from DataPreprocessing.WorldStatusEncoder import MostChangesWorldStatusEncoder
from DataPreprocessing.ObjectEncoder import ObjectEncoder

object_store = MetadataObjectStore("../../ai2thor-hugo/objects/")
change_detector = ChangeDetector()

object_encoder = ObjectEncoder(object_embedding_size=128)
world_status_encoder = MostChangesWorldStatusEncoder(object_encoder, number_of_significant_objects=3)

#dataset_analyzer = DatasetAnalyzer(object_store, change_detector)
#dataset_analyzer.plot_dataset_info()


for file_path in tqdm(object_store.list_files()):
    obj = object_store.load(file_path)
    #action_changes = change_detector.find_changes_in_file(obj)
    encoding = world_status_encoder.encode(obj)

    print(encoding)

