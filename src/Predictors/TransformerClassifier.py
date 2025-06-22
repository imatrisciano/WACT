import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding for the transformer.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Register as a buffer so it's part of the model state but not trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape (sequence_length, batch_size, d_model)
        """
        # Add positional encoding to the input embeddings
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    A transformer-based classifier for sequences of float vectors.
    """
    def __init__(self, input_dim: int, num_vectors: int, d_model: int, nhead: int,
                 num_encoder_layers: int, dim_feedforward: int, dropout: float,
                 num_classes: int):
        super().__init__()

        self.input_dim = input_dim          # Dimension of each individual input vector (e.g., 37)
        self.num_vectors = num_vectors      # Number of vectors in the input sequence (e.g., 10)
        self.d_model = d_model              # Dimension of the model's internal representation
        self.num_classes = num_classes      # Number of output classes for classification

        # 1. Input Embedding Layer: Projects each 37-dim vector to d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # 2. Positional Encoding: Adds information about the position of each vector in the sequence
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=num_vectors)

        # 3. Transformer Encoder: Core of the transformer model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False # Input expects (sequence_length, batch_size, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

        # 4. Classification Head: Maps the transformer's output to class logits
        # We'll average the sequence output of the transformer for classification
        self.classifier_head = nn.Linear(d_model, num_classes)

        self.init_weights()

    def init_weights(self):
        """Initializes weights for better training stability."""
        init_range = 0.1
        self.input_projection.weight.data.uniform_(-init_range, init_range)
        self.classifier_head.bias.data.zero_()
        self.classifier_head.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer Classifier.

        Args:
            src: Input tensor of shape (batch_size, num_vectors, input_dim).
                 For example, (batch_size, 6, 37).

        Returns:
            Output logits tensor of shape (batch_size, num_classes).
        """
        # Ensure input dimensions match expectations
        if src.shape[1] != self.num_vectors or src.shape[2] != self.input_dim:
            raise ValueError(
                f"Input tensor shape {src.shape} does not match expected "
                f"(batch_size, {self.num_vectors}, {self.input_dim})"
            )

        # Reshape for input_projection: (batch_size * num_vectors, input_dim)
        # Apply input projection to each 37-dim vector
        batch_size = src.shape[0]
        src_projected = self.input_projection(src.view(-1, self.input_dim)) # (batch_size * num_vectors, d_model)

        # Reshape back to (sequence_length, batch_size, d_model) for transformer
        src_projected = src_projected.view(batch_size, self.num_vectors, self.d_model).permute(1, 0, 2)
        # (num_vectors, batch_size, d_model)

        # Add positional encoding
        src_with_pos = self.positional_encoding(src_projected)

        # Pass through transformer encoder
        # Output shape: (num_vectors, batch_size, d_model)
        transformer_output = self.transformer_encoder(src_with_pos)

        # For classification, we can average the output across the sequence dimension
        # Or take the output of a specific token (e.g., a [CLS] token if added)
        # Here, we'll average the sequence outputs for simplicity and effectiveness.
        # Mean across the sequence dimension (dim=0)
        # Shape: (batch_size, d_model)
        avg_output = transformer_output.mean(dim=0)

        # Pass through the classification head
        logits = self.classifier_head(avg_output) # (batch_size, num_classes)

        return logits
