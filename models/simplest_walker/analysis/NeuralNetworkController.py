"""
Neural network training module for the simplest walker feedback controller.

This module implements a neural network using PyTorch to learn the feedback control
policy for the simplest walker model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, List, Dict

from models.simplest_walker.analysis.utils import (
    load_limit_cycle_solutions,
    load_linear_analysis_data,
    extract_gain_matrices,
    generate_nn_data
)


class NeuralNetworkController(nn.Module):
    """Neural network model for feedback control."""
    
    def __init__(self, input_size: int = 4, hidden_size: int = 64, output_size: int = 3, n_hidden_layers: int = 2):
        """
        Initialize the neural network model.
        
        Args:
            input_size: Number of input features (default: 4)
            hidden_size: Number of neurons in hidden layers (default: 64)
            output_size: Number of output features (default: 3)
            n_hidden_layers: Number of hidden layers (default: 2)
        """
        super().__init__()
        
        # Build layers dynamically
        layers = [nn.Linear(input_size, hidden_size), nn.Sigmoid()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Sigmoid()])
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier/Glorot initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


def prepare_data(
    n_samples: int = 500000,
    val_percent: float = 0.2,
    test_percent: float = 0.1
) -> Tuple[Dict[str, torch.Tensor], int, int]:
    """
    Prepare training, validation and test data.
    
    Args:
        n_samples: Total number of data points
        val_percent: Percentage of data for validation
        test_percent: Percentage of data for testing
        
    Returns:
        Dictionary containing data tensors and sizes of validation and test sets
    """
    # Load required data
    solutions, target_step_lengths, target_step_frequencies = load_limit_cycle_solutions()
    linear_analysis_data = load_linear_analysis_data()
    kppName = linear_analysis_data.files[0]
    kpp = linear_analysis_data[kppName]
    all_K = extract_gain_matrices(kpp)
    
    # Generate training data
    X, Y = generate_nn_data(
        solutions, target_step_lengths, target_step_frequencies,
        all_K, n_samples, sigma=0.1)
    
    # Calculate normalization parameters using entire dataset
    # X_mean = np.mean(X, axis=0).astype(np.float32)
    # X_std = np.std(X, axis=0).astype(np.float32)
    # Y_mean = np.mean(Y, axis=0).astype(np.float32)
    # Y_std = np.std(Y, axis=0).astype(np.float32)
    
    # Calculate split sizes
    n_test = int(n_samples * test_percent)
    n_val = int(n_samples * val_percent)
    n_train = n_samples - n_test - n_val
    
    # Split data and convert to float32 tensors
    X_train = torch.FloatTensor(X[:n_train].astype(np.float32))
    Y_train = torch.FloatTensor(Y[:n_train].astype(np.float32))
    
    X_val = torch.FloatTensor(X[n_train:n_train+n_val].astype(np.float32))
    Y_val = torch.FloatTensor(Y[n_train:n_train+n_val].astype(np.float32))
    
    X_test = torch.FloatTensor(X[n_train+n_val:].astype(np.float32))
    Y_test = torch.FloatTensor(Y[n_train+n_val:].astype(np.float32))
    
    # Normalize data using standardization
    # X_train = (X_train - X_mean) / X_std
    # X_val = (X_val - X_mean) / X_std
    # X_test = (X_test - X_mean) / X_std
    
    # Y_train = (Y_train - Y_mean) / Y_std
    # Y_val = (Y_val - Y_mean) / Y_std
    # Y_test = (Y_test - Y_mean) / Y_std
    
    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_val': X_val,
        'Y_val': Y_val,
        'X_test': X_test,
        'Y_test': Y_test,
        # 'X_mean': torch.FloatTensor(X_mean),
        # 'X_std': torch.FloatTensor(X_std),
        # 'Y_mean': torch.FloatTensor(Y_mean),
        # 'Y_std': torch.FloatTensor(Y_std)
    }, n_val, n_test


def train_model(
    model: nn.Module,
    data: Dict[str, torch.Tensor],
    batch_size: int = 256,
    n_epochs: int = 500,
    learning_rate: float = 0.1,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[nn.Module, List[float]]:
    """Train the neural network model."""
    model = model.to(device)
    
    # Create data loader
    train_dataset = TensorDataset(data['X_train'], data['Y_train'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9, last_epoch=-1)
    criterion = nn.MSELoss()
    
    # Training loop
    val_losses = []
    train_losses = []
    for epoch in range(n_epochs):
        # Training
        model.train() # set model to training mode
        epoch_train_loss = 0.0
        n_batches = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad() # zero gradients
            loss = criterion(model(X_batch), Y_batch) # calculate loss
            loss.backward() # backpropagate
            optimizer.step() # update weights
            epoch_train_loss += loss.item()
            n_batches += 1
        
        # Record average training loss for this epoch
        train_losses.append(epoch_train_loss / n_batches) # average loss over all batches
        
        # Validation
        model.eval() # set model to evaluation mode
        with torch.no_grad():
            val_loss = criterion(
                model(data['X_val'].to(device)),
                data['Y_val'].to(device)
            ).item() # .item() converts the loss to a python float
            val_losses.append(val_loss)
            
            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}], Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_loss:.4f}')
        
        scheduler.step()
    
    return model, val_losses, train_losses

def evaluate_model(
    model: nn.Module,
    data: Dict[str, torch.Tensor],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model on test data and return predictions in original units.
    
    Args:
        model: The trained neural network model
        data: Dictionary containing test data and normalization parameters
        device: Device to run evaluation on
        
    Returns:
        Tuple containing:
            - test_loss: Mean squared error on test set in original units
            - predictions: Model predictions in original units
            - targets: Ground truth values in original units
    """
    model.eval()
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        # Get predictions directly (no normalization)
        predictions = model(data['X_test'].to(device))
        targets = data['Y_test']
        
        # Calculate loss on unnormalized values
        test_loss = criterion(predictions, targets).item()
    
    return test_loss, predictions.cpu().numpy(), targets.cpu().numpy()