"""
Script to run step response analysis for the simplest walker model.

This script generates and analyzes step response paths for different perturbations
in step length (SL) or step frequency (SF).
"""

import argparse
import numpy as np
from models.simplest_walker.analysis.tracking_analysis import TrackingAnalysis
import torch
import os
from control import step_info

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run step response analysis for simplest walker')
    
    # Target point arguments
    parser.add_argument('--target_sl', type=float, default=0.35,
                      help='Target step length')
    parser.add_argument('--target_sf', type=float, default=0.35,
                      help='Target step frequency')
    
    # Perturbation arguments
    parser.add_argument('--perturbation_target', type=str, choices=['SL', 'SF'],
                      default='SL', help='Which parameter to perturb (SL or SF)')
    parser.add_argument('--perturbation_direction', type=str, 
                      choices=['positive', 'negative'], default='positive',
                      help='Direction of perturbation')
    parser.add_argument('--mpt', type=float, default=0.03,
                      help='Maximum tolerable perturbation in step length')
    
    
    # Analysis parameters
    parser.add_argument('--n_steps', type=int, default=20,
                      help='Number of steps in the path')
    
    return parser.parse_args()

def main():
    """Main function to run step response analysis."""
    args = parse_args()
    
    # Initialize tracking analysis
    tracking_analysis = TrackingAnalysis()
    
    # Set up target point and MPT
    target_point = (args.target_sl, args.target_sf)
    MPT = args.mpt
    
    # Generate step response path
    path, path_ks_all = tracking_analysis.generate_step_response_path(
        target_point=target_point,
        MPT=MPT,
        perturbation_target=args.perturbation_target,
        perturbation_direction=args.perturbation_direction,
        n_steps=args.n_steps
    )
    
    # Track the path
    results = tracking_analysis.track_path(
        path=path,
        path_ks_all=path_ks_all,
    )

    # Calculate step response characteristics
    if args.perturbation_target == 'SL':
        values = results['step_lengths']  # Actual step length values
        target = args.target_sl
    else:  # SF
        values = 1/results['step_times']  # Convert step times to frequencies
        target = args.target_sf
    
    # Calculate step response info using control.step_info
    step_info_dict = step_info(values, T=np.arange(len(values)), SettlingTimeThreshold=0.05)

    # Generate title for plot
    title = f"Step Response - {args.perturbation_target} {args.perturbation_direction}"
    
    # Plot results
    tracking_analysis.plot_results(path, results, title=title)
    
    # Print summary statistics
    print("\nStep Response Analysis Summary:")
    print(f"Target Point: SL={target_point[0]:.3f}, SF={target_point[1]:.3f}")
    print(f"Perturbation: {args.perturbation_target} {args.perturbation_direction}")
    print(f"MPT: {MPT:.3f}")
    
    print(f"\nStep Response Analysis:")
    print(f"Settling time: {step_info_dict['SettlingTime']:.2f} steps")
    print(f"Rise time: {step_info_dict['RiseTime']:.2f} steps")
    print(f"Peak time: {step_info_dict['PeakTime']:.2f} steps")
    print(f"Peak: {step_info_dict['Peak']:.4f}")
    print(f"Overshoot: {step_info_dict['Overshoot']:.2f}%")
    print(f"Steady-state value: {step_info_dict['SteadyStateValue']:.4f} (target: {target:.4f})")

if __name__ == "__main__":
    main() 