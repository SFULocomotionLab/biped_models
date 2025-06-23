"""
Entry point for training the feedback controller neural network.
"""

import torch
import numpy as np
from models.simplest_walker.analysis.NeuralNetworkController import NeuralNetworkController, prepare_data, train_model, evaluate_model
import matplotlib.pyplot as plt


def main():
    """Train the feedback controller neural network."""
    # Prepare data
    data, n_val, n_test = prepare_data(n_samples=1000000, val_percent=0.2, test_percent=0.1)

    # Initialize model with configurable number of hidden layers
    model = NeuralNetworkController(n_hidden_layers=2)
    
    # Train model
    trained_model, val_losses, train_losses = train_model(
        model, 
        data,
        batch_size=256,  # Reduced from 128
        n_epochs=500,
        learning_rate=0.1,  # Reduced from 0.1
    )

    # Evaluate on test set
    test_loss, predictions, targets = evaluate_model(trained_model, data)
    print(f"\nTest Loss: {test_loss:.4f}")
    
    # Print some example predictions vs targets
    print("\nExample predictions vs targets (first 5 samples):")
    print("Predictions (first 5):")
    print(predictions[:5])
    print("\nTargets (first 5):")
    print(targets[:5])
    
    # Calculate and print mean absolute error
    mae = np.mean(np.abs(predictions - targets))
    print(f"\nMean Absolute Error: {mae:.4f}")
    
    # Save model
    torch.save(trained_model.state_dict(), 'data/simplest_walker/NN_controller.pth')

    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main() 