import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm

from src.ObjectStore.MetadataObjectStore import MetadataObjectStore
from src.Predictors.MyDataset import MyDataset
from src.Predictors.TransformerClassifier import TransformerClassifier


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: torch.device
):
    """
    Trains the transformer classifier.

    Args:
        model: The TransformerClassifier instance.
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
        optimizer: Optimizer instance (e.g., Adam).
        criterion: Loss function (e.g., CrossEntropyLoss).
        num_epochs: Number of training epochs.
        device: Device to run training on ('cpu' or 'cuda').
    """
    print(f"Starting training on device: {device}")
    model.to(device)
    print("Model moved to device")

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # Clear gradients
            outputs = model(data)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update model parameters

            total_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct_predictions / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}] completed.")
        print(f"  Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_total_loss = 0
        val_correct_predictions = 0
        val_total_samples = 0

        with torch.no_grad(): # Disable gradient calculation for validation
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()

        avg_val_loss = val_total_loss / len(val_loader)
        val_accuracy = 100 * val_correct_predictions / val_total_samples
        print(f"  Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")

    print("Training finished!")


def split_dataset(whole_dataset, train_split_ratio: float, val_split_ratio: float, shuffle_dataset: bool = True):
    """
       Splits a dataset into training, validation, and test sets.

       Args:
           whole_dataset (torch.utils.data.Dataset): The entire dataset to split.
           train_split_ratio (float): The proportion of the dataset to allocate to the training set.
                                      Must be between 0 and 1.
           val_split_ratio (float): The proportion of the dataset to allocate to the validation set.
                                    Must be between 0 and 1.
                                    The test set will take the remaining proportion.
           shuffle_dataset (bool): Whether to shuffle the dataset indices before splitting.

       Returns:
           tuple: A tuple containing (train_loader, validation_loader, test_loader).
       """

    if not (0 < train_split_ratio < 1 and 0 <= val_split_ratio < 1):
        raise ValueError("train_split_ratio and val_split_ratio must be between 0 and 1.")
    if train_split_ratio + val_split_ratio >= 1:
        raise ValueError("The sum of train_split_ratio and val_split_ratio must be less than 1 "
                         "to leave room for a test set.")

    dataset_size = len(whole_dataset)
    indices = list(range(dataset_size))

    if shuffle_dataset:
        np.random.shuffle(indices)

    # Calculate split points
    train_split_point = int(np.floor(train_split_ratio * dataset_size))
    val_split_point = int(np.floor((train_split_ratio + val_split_ratio) * dataset_size))

    # Split indices
    train_indices = indices[:train_split_point]
    val_indices = indices[train_split_point:val_split_point]
    test_indices = indices[val_split_point:]

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create DataLoaders (assuming BATCH_SIZE is defined in the scope where this function is called)
    # If BATCH_SIZE is not global, you should pass it as an argument to the function.
    try:
        train_loader = torch.utils.data.DataLoader(whole_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(whole_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)
        test_loader = torch.utils.data.DataLoader(whole_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)
    except NameError:
        raise NameError("BATCH_SIZE is not defined. Please define it or pass it as an argument to split_dataset.")

    return train_loader, validation_loader, test_loader

# --- Main Execution ---
if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Input size, classes count
    NUMBER_OF_ACTION_SIGNIFICANT_OBJECTS = 10 # How many objects to consider when analyzing an executed action
    INPUT_DIM = 37            # Length of each object encoding vector
    NUM_VECTORS = 2 * NUMBER_OF_ACTION_SIGNIFICANT_OBJECTS           # Number of vectors in the input sequence = 2*number_of_significant_objects since we store both before and after action objects
    NUM_CLASSES = 13          # Number of possible output classes (for softmax)

    # Model parameters
    D_MODEL = 32             # Embedding dimension for the transformer
    NHEAD = 4                 # Number of attention heads (must divide D_MODEL)
    NUM_ENCODER_LAYERS = 3    # Number of transformer encoder layers
    DIM_FEEDFORWARD = 512     # Dimension of the feedforward network in transformer
    DROPOUT = 0.1             # Dropout rate

    # Learning parameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    NUM_EPOCHS = 40           # Number of training epochs

    # Initialize Dataset and DataLoader
    print("Creating datasets...")
    object_store = MetadataObjectStore("../../../ai2thor-hugo/objects/")
    whole_dataset = MyDataset(object_store, NUMBER_OF_ACTION_SIGNIFICANT_OBJECTS)

    np.random.seed(42)

    # Test size: 10%
    # Validation size: 20%
    # Training size: 70%
    train_loader, validation_loader, test_loader = split_dataset(whole_dataset, 0.7, 0.2, shuffle_dataset=True)

    # Initialize Model, Loss, and Optimizer
    print("Initializing model, loss function, and optimizer...")
    model = TransformerClassifier(
        input_dim=INPUT_DIM,
        num_vectors=NUM_VECTORS,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES
    )

    criterion = nn.CrossEntropyLoss() # Includes softmax implicitly
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Start training
    print("Starting training process...")
    train_classifier(model, train_loader, validation_loader, optimizer, criterion, NUM_EPOCHS, device)

    # --- Example Inference (after training) ---
    print("\n--- Example Inference ---")
    model.eval() # Set model to evaluation mode
    # Create a dummy input for inference
    dummy_input = torch.randn(1, NUM_VECTORS, INPUT_DIM).to(device) # One sample
    with torch.no_grad():
        output_logits = model(dummy_input)
        probabilities = torch.softmax(output_logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Output logits: {output_logits.cpu().numpy()}")
    print(f"Output probabilities: {probabilities.cpu().numpy()}")
    print(f"Predicted class index: {predicted_class}")

    torch.save(model, "predictor.pth")

