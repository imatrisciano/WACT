from ObjectStore.MetadataObjectStore import MetadataObjectStore
from DataPreprocessing.ChangeDetector import ChangeDetector
from DataAnalysis.DatasetAnalysis import DatasetAnalyzer

object_store = MetadataObjectStore("../../ai2thor-hugo/objects/")
change_detector = ChangeDetector()

dataset_analyzer = DatasetAnalyzer(object_store, change_detector)
dataset_analyzer.plot_dataset_info()


#for file_path in tqdm(object_store.list_files()):
    #obj = object_store.load(file_path)
    #action_changes = change_detector.find_changes_in_file(obj)

    #print(action_changes)

