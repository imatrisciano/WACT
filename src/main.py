
from ObjectStore.MetadataObjectStore import MetadataObjectStore
from DataPreprocessing.ChangeDetector import ChangeDetector

object_store = MetadataObjectStore("../data/action_effects/")
change_detector = ChangeDetector()

obj = object_store.load("scene_18_2.json")


action_changes = change_detector.find_changes_in_file(obj)
print(action_changes)

