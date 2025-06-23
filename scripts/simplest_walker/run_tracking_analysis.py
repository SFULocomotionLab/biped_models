#!/usr/bin/env python3
"""
Entry point script for running tracking analysis on the simplest walker model.

This script provides a command-line interface to run tracking analysis with different
path types and parameters.
"""

import argparse
import numpy as np
from models.simplest_walker.analysis.tracking_analysis import TrackingAnalysis
import matplotlib.pyplot as plt
import torch
from models.simplest_walker.analysis.NeuralNetworkController import NeuralNetworkController

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run tracking analysis on the simplest walker model'
    )
    
    # Starting point
    parser.add_argument(
        '--start-point',
        type=float,
        nargs=2,
        default=[0.6, 0.6],
        help='Starting point [step_length, step_frequency]'
    )
    
    # Sine path parameters
    parser.add_argument(
        '--sine-params',
        type=float,
        nargs=6,
        default=[0.1, 0.1, 2.0, 1.0, 1.0, 31],
        help='Sine parameters [ampSL, ampSF, freqSL, freqSF, time, n_steps]'
    )
    
    return parser.parse_args()

def main():
    """Main function to run tracking analysis."""
    args = parse_args()
    
    # Initialize tracking analysis
    tracking = TrackingAnalysis()
    
    # Generate path
    sine_params = {
        'ampSL': args.sine_params[0],
        'ampSF': args.sine_params[1],
        'freqSL': args.sine_params[2],
        'freqSF': args.sine_params[3],
        'time': args.sine_params[4],
        'n_steps': int(args.sine_params[5])
    }
    path, path_ks_all = tracking.generate_sinusoid_path(
        tuple(args.start_point), sine_params
    )
    
    # Load the nn_controller trained model
    nn_model = NeuralNetworkController(input_size=4, hidden_size=64, output_size=3, n_hidden_layers=2)
    nn_model.load_state_dict(torch.load('data/simplest_walker/NN_controller.pth'))
    
    # Track the path
    results = tracking.track_path(path, path_ks_all, nn_model)
    
    # Print results
    print("\nTracking Results:")
    print(f"Total steps: {len(results['step_lengths'])}")
    print(f"Average reward: {np.mean(results['rewards']):.3f}")
    print(f"RMS error SL: {np.sqrt(np.mean(results['rmse_sl']**2)):.3f}")
    print(f"RMS error SF: {np.sqrt(np.mean(results['rmse_sf']**2)):.3f}")
    
    # Plot results
    tracking.plot_results(path, results)
    
    # Show plots
    plt.show()

if __name__ == '__main__':
    main() 