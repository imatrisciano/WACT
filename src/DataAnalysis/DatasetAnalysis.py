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

            for changed_object in action_effect.object_changes:
                for changed_property in changed_object.object_changes:
                    self.changed_property_count[changed_property.property_path_str()] += 1

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
        print("Number of changes properties per each action:")
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


        print("Number of changes properties per each action, normalized per number of changed objects:")
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

        self._plot_action_names_statistics()

        self._plot_number_of_changes_statistics()

        self._plot_changed_property_count()




