import os.path
from typing import Literal

import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torchinfo import summary
from torch.utils.data import DataLoader

from src.Predictors.MyDataset import MyDataset
from src.Predictors.TransformerClassifier import TransformerClassifier

class ClassifierManager:
    def __init__(self, object_store, dataset: MyDataset, number_of_significant_objects = 10, model_save_path: str = "predictor.pth", device = None):
        self.object_store = object_store

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.whole_dataset = dataset

        self.train_action_losses = []
        self.train_object_losses = []
        self.train_action_accuracies = []
        self.train_object_accuracies = []
        self.val_action_losses = []
        self.val_object_losses = []
        self.val_action_accuracies = []
        self.val_object_accuracies = []

        self.model: nn.Module = None
        self.action_criterion: nn.Module = None # Loss function for the action classifier (e.g., CrossEntropyLoss).
        self.object_criterion: nn.Module = None # Loss function for the object classifier
        self.optimizer: optim.Optimizer = None # Optimizer instance (e.g., Adam).

        # Input size, classes count
        self.NUMBER_OF_ACTION_SIGNIFICANT_OBJECTS = number_of_significant_objects # How many objects to consider when analyzing an executed action
        self.INPUT_DIM = 37            # Length of each object encoding vector
        self.NUM_VECTORS = 2 * self.NUMBER_OF_ACTION_SIGNIFICANT_OBJECTS           # Number of vectors in the input sequence = 2*number_of_significant_objects since we store both before and after action objects
        self.NUM_ACTION_CLASSES = 14          # Number of possible output classes (for softmax)
        self.NUM_OBJECT_CLASSES = self.NUMBER_OF_ACTION_SIGNIFICANT_OBJECTS

        # Model parameters
        self.D_MODEL = 32             # Embedding dimension for the transformer
        self.NHEAD = 2                 # Number of attention heads (must divide D_MODEL)
        self.NUM_ENCODER_LAYERS = 4    # Number of transformer encoder layers
        self.DIM_FEEDFORWARD = 512     # Dimension of the feedforward network in transformer
        self.DROPOUT = 0.1             # Dropout rate

        # Learning parameters
        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 256
        self.NUM_EPOCHS = 150           # Number of training epochs

        self.model_save_path = model_save_path

    def _train_classifier(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
        ):
        """
        Trains the transformer classifier.

        Args:
            train_loader: DataLoader for the training data.
            val_loader: DataLoader for the validation data.
        """

        print(f"Starting training on device: {self.device}")
        self.model.to(self.device)
        print("Model moved to device")

        print(f"Training for {self.NUM_EPOCHS} epochs...")
        for epoch in range(self.NUM_EPOCHS):
            avg_action_train_loss, train_action_accuracy, avg_object_train_loss, train_object_accuracy = (
                self._train_epoch(epoch, train_loader))
            val_action_loss, val_action_accuracy,\
                val_object_loss, val_object_accuracy = self._compute_validation_loss_and_accuracy(val_loader)


            self.val_action_losses.append(val_action_loss)
            self.val_action_accuracies.append(val_action_accuracy)
            self.val_object_losses.append(val_object_loss)
            self.val_object_accuracies.append(val_object_accuracy)

            #print(f"  Validation Action Loss: {val_action_loss:.4f}, Accuracy: {val_action_accuracy:.3f}%")
            #print(f"  Validation Object Loss: {val_object_loss:.4f}, Accuracy: {val_object_accuracy:.3f}%\n")

            print(f"[EPOCH]: {epoch + 1}/{self.NUM_EPOCHS} "
                  f"\t[Action TRAIN ACC]: {train_action_accuracy:.4f}%"
                  f"\t[Action VAL ACC]: {val_action_accuracy:.4f}%"
                  f"\t[Object TRAIN ACC]: {train_object_accuracy:.4f}%"
                  f"\t[Object VAL ACC]: {val_object_accuracy:.4f}%")

        print("Training finished!")

    def _train_epoch(self, epoch, train_loader) -> (float, float, float, float):
        """
        Trains the model for one epoch.
        :param epoch: the current epoch number
        :param train_loader: the training set loader
        :return: avg_action_train_loss, train_action_accuracy, avg_object_train_loss, train_object_accuracy
        """
        self.model.train()  # Set model to training mode

        total_action_loss = 0
        total_object_loss = 0
        correct_action_predictions = 0
        correct_object_predictions = 0
        total_action_samples = 0
        total_object_samples = 0

        for batch_idx, (data, action_labels, object_labels) in enumerate(train_loader):
            data = data.to(self.device)
            action_labels = action_labels.to(self.device)
            object_labels = object_labels.to(self.device)

            self.optimizer.zero_grad()  # Clear gradients
            action_outputs, object_outputs = self.model(data)  # Forward pass

            # Calculate individual losses
            action_loss = self.action_criterion(action_outputs, action_labels)
            object_loss = self.object_criterion(object_outputs, object_labels)

            # Compute total loss with a weighted sum
            action_importance = 1.0
            object_importance = 1.0
            total_loss = action_importance * action_loss + object_importance * object_loss

            total_loss.backward()  # Backward pass (compute gradients)
            self.optimizer.step()  # Update model parameters

            total_action_loss += action_loss.item()
            total_object_loss += object_loss.item()

            # Calculate training accuracy for the Action classifier
            _, predicted_task1 = torch.max(action_outputs.data, 1)
            total_action_samples += action_labels.size(0)
            correct_action_predictions += (predicted_task1 == action_labels).sum().item()

            # Calculate training accuracy for the Object classifier
            _, predicted_task2 = torch.max(object_outputs.data, 1)
            total_object_samples += object_labels.size(0)
            correct_object_predictions += (predicted_task2 == object_labels).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch [{epoch + 1}/{self.NUM_EPOCHS}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                      f"Total Loss: {total_loss.item():.4f}, "
                      f"Action loss: {action_loss.item():.4f}, Object loss: {object_loss.item():.4f}")

        avg_action_train_loss = total_action_loss / len(train_loader)
        train_action_accuracy = 100 * correct_action_predictions / total_action_samples

        avg_object_train_loss = total_object_loss / len(train_loader)
        train_object_accuracy = 100 * correct_object_predictions / total_object_samples

        self.train_action_losses.append(avg_action_train_loss)
        self.train_action_accuracies.append(train_action_accuracy)
        self.train_object_losses.append(avg_object_train_loss)
        self.train_object_accuracies.append(train_object_accuracy)

        #print(f"Epoch [{epoch + 1}/{self.NUM_EPOCHS}] completed.")
        #print(f"  Training Action Loss: {avg_action_train_loss:.4f}, Accuracy: {train_action_accuracy:.2f}%")
        #print(f"  Training Object Loss: {avg_object_train_loss:.4f}, Accuracy: {train_object_accuracy:.2f}%")

        return avg_action_train_loss, train_action_accuracy, avg_object_train_loss, train_object_accuracy


    def _compute_validation_loss_and_accuracy(self, data_loader) -> (float, float):
        """
        Computes the model's loss and accuracy on the given data_loader
        :param data_loader: Provider of the validation data
        :return: model's average loss on the data, model's accuracy on the data
        """

        # Validation phase
        self.model.eval()  # Set model to evaluation mode
        val_total_action_loss = 0
        val_correct_action_predictions = 0
        val_total_action_samples = 0

        val_total_object_loss = 0
        val_correct_object_predictions = 0
        val_total_object_samples = 0

        with torch.no_grad():  # Disable gradient calculation for validation
            for data, action_labels, object_labels in data_loader:
                data, action_labels, object_labels = data.to(self.device), action_labels.to(self.device), object_labels.to(self.device)

                action_outputs, object_outputs = self.model(data)

                action_loss = self.action_criterion(action_outputs, action_labels)
                object_loss = self.object_criterion(object_outputs, object_labels)
                val_total_action_loss += action_loss.item()
                val_total_object_loss += object_loss.item()

                _, predicted_action = torch.max(action_outputs.data, 1)
                val_total_action_samples += action_labels.size(0)
                val_correct_action_predictions += (predicted_action == action_labels).sum().item()

                _, predicted_object = torch.max(object_outputs.data, 1)
                val_total_object_samples += object_labels.size(0)
                val_correct_object_predictions += (predicted_object == object_labels).sum().item()

        avg_action_loss = val_total_action_loss / len(data_loader)
        action_accuracy = 100 * val_correct_action_predictions / val_total_action_samples

        avg_object_loss = val_total_object_loss / len(data_loader)
        object_accuracy = 100 * val_correct_object_predictions / val_total_object_samples

        return avg_action_loss, action_accuracy, avg_object_loss, object_accuracy

    def _validate_model(self, data_loader, task: Literal["action", "object"]) -> (float, float):
        """
        Computes the model's loss and accuracy on the given data_loader and builds and shows the confusion matrix
        :param data_loader: Provider of the validation data
        :return: model's average loss on the data, model's accuracy on the data
        """

        # Validation phase
        self.model.eval()  # Set model to evaluation mode
        val_total_loss = 0
        val_correct_predictions = 0
        val_total_samples = 0

        all_labels = []  # To store all true labels
        all_predictions = []  # To store all predicted labels

        with torch.no_grad():  # Disable gradient calculation for validation
            for data, action_labels, object_labels in data_loader:
                data, action_labels, object_labels = data.to(self.device), action_labels.to(self.device), object_labels.to(self.device)

                action_outputs, object_outputs = self.model(data)

                if task == "action":
                    labels = action_labels
                    outputs = action_outputs
                    criterion = self.action_criterion
                elif task == "object":
                    labels = object_labels
                    outputs = object_outputs
                    criterion = self.object_criterion
                else:
                    raise NotImplementedError

                loss = criterion(outputs, labels)
                val_total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()

                # Collect labels and predictions for confusion matrix
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Build and plot confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        if task == "action":
            class_names = self.whole_dataset.get_labels()
            class_names.remove("Unknown")  # removes the first, unused entry
        elif task == "object":
            class_names = list(range(0, self.NUM_OBJECT_CLASSES))
        else:
            raise NotImplementedError

        plt.figure(figsize=(11, 10))
        plt.subplots_adjust(left=0.17, bottom=0.17, right=1, top=0.95) # better centering of the image, empirical values
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix')
        plt.show()

        avg_loss = val_total_loss / len(data_loader)
        accuracy = 100 * val_correct_predictions / val_total_samples

        return avg_loss, accuracy

    def plot_training_graphs(self):
        font_size = 12

        # Plot the Graph Loss Curves
        plt.figure(figsize=[8, 6])
        plt.plot(self.train_action_losses, 'r', linewidth=2.0)
        plt.plot(self.val_action_losses, 'b', linewidth=2.0)
        plt.legend(['Training loss', 'Validation Loss'], fontsize=font_size)
        plt.xlabel('Epochs ', fontsize=font_size)
        plt.ylabel('Loss', fontsize=font_size)
        plt.title('Loss Curves - Action classifier', fontsize=font_size)

        # Accuracy Curves
        plt.figure(figsize=[8, 6])
        plt.plot(self.train_action_accuracies, 'r', linewidth=2.0)
        plt.plot(self.val_action_accuracies, 'b', linewidth=2.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=font_size)
        plt.xlabel('Epochs ', fontsize=font_size)
        plt.ylabel('Accuracy', fontsize=font_size)
        plt.title('Accuracy Curves - Action classifier', fontsize=font_size)

        plt.figure(figsize=[8, 6])
        plt.plot(self.train_object_losses, 'r', linewidth=2.0)
        plt.plot(self.val_object_losses, 'b', linewidth=2.0)
        plt.legend(['Training loss', 'Validation Loss'], fontsize=font_size)
        plt.xlabel('Epochs ', fontsize=font_size)
        plt.ylabel('Loss', fontsize=font_size)
        plt.title('Loss Curves - Object classifier', fontsize=font_size)

        plt.figure(figsize=[8, 6])
        plt.plot(self.train_object_accuracies, 'r', linewidth=2.0)
        plt.plot(self.val_object_accuracies, 'b', linewidth=2.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=font_size)
        plt.xlabel('Epochs ', fontsize=font_size)
        plt.ylabel('Accuracy', fontsize=font_size)
        plt.title('Accuracy Curves - Object classifier', fontsize=font_size)

        plt.show()

    def train(self):
        # Device configuration
        print(f"Using device: {self.device}")

        # Initialize Dataset and DataLoader
        print("Creating datasets...")
        self.whole_dataset.load()

        np.random.seed(42)

        # Test size: 10%
        # Validation size: 20%
        # Training size: 70%
        train_loader, validation_loader, test_loader = self.whole_dataset.split_dataset(
            train_split_ratio=0.7,
            val_split_ratio=0.2,
            batch_size=self.BATCH_SIZE,
            shuffle_dataset=True)

        # Initialize Model, Loss, and Optimizer
        print("Initializing model, loss function, and optimizer...")
        self.model = TransformerClassifier(
            input_dim=self.INPUT_DIM,
            num_vectors=self.NUM_VECTORS,
            d_model=self.D_MODEL,
            nhead=self.NHEAD,
            num_encoder_layers=self.NUM_ENCODER_LAYERS,
            dim_feedforward=self.DIM_FEEDFORWARD,
            dropout=self.DROPOUT,
            num_classes_task1=self.NUM_ACTION_CLASSES,
            num_classes_task2=self.NUM_OBJECT_CLASSES
        )

        self.action_criterion = nn.CrossEntropyLoss() # Includes softmax implicitly
        self.object_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

        # Start training
        print("Starting training process...")
        self._train_classifier(train_loader, validation_loader)

        print("Evaluating test accuracy for Action prediction...")
        test_action_loss, test_action_accuracy = self._validate_model(test_loader, task="action")
        print(f"Test Loss for Action Prediction: {test_action_loss:.4f}, Test Accuracy for Action Prediction: {test_action_accuracy:.3f}%\n")

        print("Evaluating test accuracy for Object prediction...")
        test_object_loss, test_object_accuracy = self._validate_model(test_loader, task="object")
        print(f"Test Loss for Object Prediction: {test_object_loss:.4f}, Test Accuracy for Object Prediction: {test_object_accuracy:.3f}%\n")

        self.plot_training_graphs()

        print("Saving model...")

        torch.save(self.model, self.model_save_path)
        print(f"Model saved of file {self.model_save_path}")

        # Print the model summary
        summary(self.model)

    def load_model(self):
        if not os.path.exists(self.model_save_path):
            raise Exception(f"Model file was not found in path {self.model_save_path}.\n"
                            f"Please run the training process or download a pre-trained model")

        print("\n\n" + "="*50)
        print(f"WARNING: loading model '{self.model_save_path}' with weights_only=False")
        print("This can be a cybersecurity risk if you don't trust whoever gave you the model file.")

        answer = input(" > Continue anyway? [y/N]: ")
        if answer.lower().strip() != "y":
            raise Exception("Untrusted model loading was canceled by the user")

        self.model = torch.load(self.model_save_path, weights_only=False)

    def inference(self, vector) -> (int, str, int):
        """
        Performs inference on a given numeric vector input
        :param vector: input data (encoded before and after action world status)
        :return: the predicted action class number, the corresponding action name, the predicted object index
        """

        if self.model is None:
            self.load_model()

        self.model.eval()
        tensor_input = torch.tensor(vector).to(self.device)
        with torch.no_grad():
            output_logits_action_classifier, output_logits_object_classifier = self.model(tensor_input)

            action_probabilities = torch.softmax(output_logits_action_classifier, dim=1)
            predicted_action_class_index = torch.argmax(action_probabilities, dim=1).item()
            predicted_action_confidence = action_probabilities[0, predicted_action_class_index].item()

            object_probabilities = torch.softmax(output_logits_object_classifier, dim=1)
            predicted_object_class_index = torch.argmax(object_probabilities, dim=1).item()
            predicted_object_confidence = object_probabilities[0, predicted_object_class_index].item()

        action_name = self.whole_dataset.id2label(predicted_action_class_index)

        return predicted_action_class_index, action_name, predicted_object_class_index, predicted_action_confidence, predicted_object_confidence


