from statistics import median, mean

import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from src.DataPreprocessing.ChangeDetector import ChangeDetector
from src.Models.ActionEffect import ActionEffect
from src.ObjectStore.MetadataObjectStore import MetadataObjectStore


class DatasetAnalyzer:
    def __init__(self, object_store: MetadataObjectStore, change_detector: ChangeDetector):
        self.object_store = object_store
        self.change_detector = change_detector
        self._dataset_analyzed: bool = False

        self.action_counts = defaultdict(int)
        self.changed_property_count = defaultdict(int)
        self.action_property_changes = []
        self.action_object_changes = []
        self.salient_materials = []
        self.object_types = []
        self.fill_liquids = []
        self.actions_names = []

    def _analyze_dataset(self):
        # Analysis is only required once
        if self._dataset_analyzed:
            return

        # Iterates over each file
        print("Reading the dataset...")
        for file_path in tqdm(self.object_store.list_files()):

            # Read and deserialize the file
            obj: dict = self.object_store.load(file_path)

            # Read action_name from the file and populate the statistics
            if 'action_name' in obj:
                self.action_counts[obj['action_name']] += 1

            # Analyze object changes
            action_effect: ActionEffect = self.change_detector.find_changes_in_file(obj)
            self.action_property_changes.append(action_effect.number_of_property_changes())
            self.action_object_changes.append(action_effect.number_of_changed_objects())

            if not action_effect.action_name in self.actions_names:
                self.actions_names.append(action_effect.action_name)

            for changed_object in action_effect.object_changes:
                for changed_property in changed_object.object_changes:
                    self.changed_property_count[changed_property.property_path_str()] += 1

            for scene_object in obj["before_world_status"]["objects"]:
                object_type = scene_object["objectType"]
                if object_type not in self.object_types:
                    self.object_types.append(object_type)

                fill_liquid = scene_object["fillLiquid"]
                if fill_liquid not in self.fill_liquids:
                    self.fill_liquids.append(fill_liquid)

                object_materials = scene_object["salientMaterials"]
                if object_materials is None:
                    continue

                for material in object_materials:
                    if material not in self.salient_materials:
                        self.salient_materials.append(material)


    def _get_action_names_and_counts(self) -> (list[str], list[int]):
        action_names = list(self.action_counts.keys())
        counts = list(self.action_counts.values())

        return action_names, counts

    def _plot_action_names_statistics(self):
        # Prepare data for plotting
        action_names, action_counts = self._get_action_names_and_counts()

        # Print action counts
        print("Action counts:")
        for name, count in zip(action_names, action_counts):
            print(f"- {name}: {count}")
        print("\n\n")

        # Create the action counts bar graph
        plt.figure(figsize=(12, 6))
        plt.bar(action_names, action_counts, color='skyblue', edgecolor='#333')
        plt.xlabel('Action Name')
        plt.ylabel('Number of Instances')
        plt.title('Distribution of Actions')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping
        plt.show()

    def _plot_number_of_changes_statistics(self):
        print("Number of changed properties per each action:")
        print(f" - min: {min(self.action_property_changes)}")
        print(f" - max: {max(self.action_property_changes)}")
        print(f" - avg: {mean(self.action_property_changes)}")
        print(f" - median: {median(self.action_property_changes)}")
        print("\n\n")

        n, bins, patches = plt.hist(self.action_property_changes, bins=40, color='skyblue', edgecolor='#333')

        plt.xlabel('Action property changes')
        plt.ylabel('Frequency')
        plt.title(f'Frequency Distribution of Action changes')
        plt.xticks(bins, rotation=90, ha='right')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.show()


        print("Number of changed objects per each action:")
        print(f" - min: {min(self.action_object_changes)}")
        print(f" - max: {max(self.action_object_changes)}")
        print(f" - avg: {mean(self.action_object_changes)}")
        print(f" - median: {median(self.action_object_changes)}")
        print("\n\n")

        print("Number of changed properties per each action, normalized per number of changed objects:")
        normalized_property_changes = [self.action_property_changes[i] / self.action_object_changes[i] for i in range(0, len(self.action_property_changes))]

        print(f" - min: {min(normalized_property_changes)}")
        print(f" - max: {max(normalized_property_changes)}")
        print(f" - avg: {mean(normalized_property_changes)}")
        print(f" - median: {median(normalized_property_changes)}")
        print("\n\n")

        n, bins, patches = plt.hist(normalized_property_changes, bins=40, color='skyblue', edgecolor='#333')

        plt.xlabel('Action property changes, normalized per number of changed objects')
        plt.ylabel('Frequency')
        plt.title(f'Frequency Distribution of Action changes')
        plt.xticks(bins, rotation=90, ha='right')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.show()

    def _plot_changed_property_count(self):
        property_names = list(self.changed_property_count.keys())
        property_counts = list(self.changed_property_count.values())

        plt.figure(figsize=(12, 6))
        plt.bar(property_names, property_counts, color='skyblue', edgecolor='#333')
        plt.xlabel('Property Path')
        plt.ylabel('Number of Instances')
        plt.title('Distribution of Property Changes')
        plt.xticks(rotation=90, ha='right')  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping
        plt.show()

    def plot_dataset_info(self):
        """
        Uses the object_store to read the dataset and gathers some statistics on that
        """

        self._analyze_dataset()

        print("Salient materials: ")
        for i, material in enumerate(self.salient_materials):
            print(f"{i+1}: \"{material}\",")
        print()

        print("Object types: ")
        for i, obj_type in enumerate(self.object_types):
            print(f"{i + 1}: \"{obj_type}\",")
        print()

        print("Fill liquids: ")
        for i, fill_liquid in enumerate(self.fill_liquids):
            print(f"{i + 1}: \"{fill_liquid}\",")
        print()

        print("Actions: ")
        for i, action in enumerate(self.actions_names):
            print(f"{i + 1}: \"{action}\",")
        print()

        self._plot_action_names_statistics()

        self._plot_number_of_changes_statistics()

        self._plot_changed_property_count()




