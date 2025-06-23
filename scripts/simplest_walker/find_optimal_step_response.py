"""
                NOT COMPLETE AND NOT WORKING YET

Entry point script for running step response analysis on the simplest walker model.

This script allows running step response analysis with configurable target points
and perturbation sizes for step length (SL) and step frequency (SF).
"""

import argparse
import numpy as np
from models.simplest_walker.analysis.optimal_step_response import StepResponseAnalysis


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run step response analysis for simplest walker')
    
    # Target point arguments
    parser.add_argument('--target-point', type=float, nargs=2, required=True,
                       help='Target point as [SL, SF] (dimensionless)')
    
    # Perturbation arguments
    parser.add_argument('--perturbation', type=float, default=0.15,
                       help='Perturbation size as fraction of target (default: 0.15)')
    parser.add_argument('--perturb-axis', type=str, choices=['SL', 'SF'], required=True,
                       help='Axis to apply perturbation (SL or SF)')
    
    # Analysis parameters
    parser.add_argument('--n-steps', type=int, default=10,
                       help='Number of steps to simulate (default: 10)')
    
    return parser.parse_args()


def main():
    """Main function to run step response analysis."""
    # Parse command line arguments
    args = parse_args()
    
    # Create step response analyzer
    analyzer = StepResponseAnalysis()
    
    # Define target point
    target_point = tuple(args.target_point)
    
    # Create step size perturbations based on perturb-axis
    if args.perturb_axis == 'SL':
        step_size = np.array([
            [1 - args.perturbation, 1],  # SL below
            [1 + args.perturbation, 1],  # SL above
        ])
    else:  # SF
        step_size = np.array([
            [1, 1 - args.perturbation],  # SF below
            [1, 1 + args.perturbation]   # SF above
        ])
    
    # Run analysis
    results = analyzer.analyze_step_response(
        target_point=target_point,
        step_size=step_size,
        n_steps=args.n_steps,
        settling_time=np.array([args.settling_time_sl, args.settling_time_sf])
    )
    
    # Plot results
    analyzer.plot_step_response(results, target_point, step_size)


if __name__ == '__main__':
    main() 