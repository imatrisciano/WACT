from tqdm import tqdm

from ObjectStore.MetadataObjectStore import MetadataObjectStore
from DataPreprocessing.ChangeDetector import ChangeDetector

object_store = MetadataObjectStore("../data/action_effects/")
change_detector = ChangeDetector()

for file_path in tqdm(object_store.list_files()):
    obj = object_store.load(file_path)
    action_changes = change_detector.find_changes_in_file(obj)

    print(action_changes)

