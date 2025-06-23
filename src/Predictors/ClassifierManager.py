import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

from src.Predictors.MyDataset import MyDataset
from src.Predictors.TransformerClassifier import TransformerClassifier

class ClassifierManager:
    def __init__(self, object_store, device = None):
        self.object_store = object_store
        self.whole_dataset = None
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        self.model: nn.Module = None
        self.criterion: nn.Module = None # Loss function (e.g., CrossEntropyLoss).
        self.optimizer: optim.Optimizer = None # Optimizer instance (e.g., Adam).

        # Input size, classes count
        self.NUMBER_OF_ACTION_SIGNIFICANT_OBJECTS = 10 # How many objects to consider when analyzing an executed action
        self.INPUT_DIM = 37            # Length of each object encoding vector
        self.NUM_VECTORS = 2 * self.NUMBER_OF_ACTION_SIGNIFICANT_OBJECTS           # Number of vectors in the input sequence = 2*number_of_significant_objects since we store both before and after action objects
        self.NUM_CLASSES = 13          # Number of possible output classes (for softmax)

        # Model parameters
        self.D_MODEL = 32             # Embedding dimension for the transformer
        self.NHEAD = 4                 # Number of attention heads (must divide D_MODEL)
        self.NUM_ENCODER_LAYERS = 3    # Number of transformer encoder layers
        self.DIM_FEEDFORWARD = 512     # Dimension of the feedforward network in transformer
        self.DROPOUT = 0.1             # Dropout rate

        # Learning parameters
        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 64
        self.NUM_EPOCHS = 10           # Number of training epochs

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

        for epoch in range(self.NUM_EPOCHS):
            self._train_epoch(epoch, train_loader)
            val_loss, val_accuracy = self._validate_model(val_loader)

            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            print(f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.3f}%\n")

        print("Training finished!")

    def _train_epoch(self, epoch, train_loader):
        self.model.train()  # Set model to training mode
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()  # Clear gradients
            outputs = self.model(data)  # Forward pass
            loss = self.criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass (compute gradients)
            self.optimizer.step()  # Update model parameters

            total_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch [{epoch + 1}/{self.NUM_EPOCHS}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct_predictions / total_samples

        self.train_losses.append(avg_train_loss)
        self.train_accuracies.append(train_accuracy)

        print(f"Epoch [{epoch + 1}/{self.NUM_EPOCHS}] completed.")
        print(f"  Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    def _validate_model(self, data_loader) -> (float, float):
        """
        Computes the model's loss and accuracy on the given data_loader
        :param data_loader: Provider of the validation data
        :return: model's average loss on the data, model's accuracy on the data
        """

        # Validation phase
        self.model.eval()  # Set model to evaluation mode
        val_total_loss = 0
        val_correct_predictions = 0
        val_total_samples = 0

        with torch.no_grad():  # Disable gradient calculation for validation
            for data, labels in data_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                val_total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()

        avg_loss = val_total_loss / len(data_loader)
        accuracy = 100 * val_correct_predictions / val_total_samples

        return avg_loss, accuracy

    def plot_training_graphs(self):
        font_size = 12

        # Plot the Graph Loss Curves
        plt.figure(figsize=[8, 6])
        plt.plot(self.train_losses, 'r', linewidth=2.0)
        plt.plot(self.val_losses, 'b', linewidth=2.0)
        plt.legend(['Training loss', 'Validation Loss'], fontsize=font_size)
        plt.xlabel('Epochs ', fontsize=font_size)
        plt.ylabel('Loss', fontsize=font_size)
        plt.title('Loss Curves', fontsize=font_size)

        # Accuracy Curves
        plt.figure(figsize=[8, 6])
        plt.plot(self.train_accuracies, 'r', linewidth=2.0)
        plt.plot(self.val_accuracies, 'b', linewidth=2.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=font_size)
        plt.xlabel('Epochs ', fontsize=font_size)
        plt.ylabel('Accuracy', fontsize=font_size)
        plt.title('Accuracy Curves', fontsize=font_size)

        plt.show()

    def train(self):
        # Device configuration
        print(f"Using device: {self.device}")

        # Initialize Dataset and DataLoader
        print("Creating datasets...")
        self.whole_dataset = MyDataset(self.object_store, self.NUMBER_OF_ACTION_SIGNIFICANT_OBJECTS)

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
            num_classes=self.NUM_CLASSES
        )

        self.criterion = nn.CrossEntropyLoss() # Includes softmax implicitly
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

        # Start training
        print("Starting training process...")
        self._train_classifier(train_loader, validation_loader)

        print("Evaluating test accuracy...")
        test_loss, test_accuracy = self._validate_model(test_loader)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.3f}%\n")

        self.plot_training_graphs()

        print("Saving model...")
        model_save_path = "predictor.pth"
        torch.save(self.model, model_save_path)
        print(f"Model saved of file {model_save_path}")

    def inference(self, vector) -> (int, str):
        """
        Performes inference on a given numeric vector input
        :param vector: input data (encoded before and after action world status)
        :return: both the predicted class number and its name
        """

        self.model.eval()  # Set model to evaluation mode
        tensor_input = torch.tensor(vector).to(self.device)
        with torch.no_grad():
            output_logits = self.model(tensor_input)
            probabilities = torch.softmax(output_logits, dim=1)
            predicted_class_index = torch.argmax(probabilities, dim=1).item()

        predicted_class_name = self.whole_dataset
        return predicted_class_index, predicted_class_name


