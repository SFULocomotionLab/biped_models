import sys
import os
import argparse
import numpy as np

from models.simplest_walker.analysis.maximum_tolerable_perturbation import MaximumTolerablePerturbation

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

#!/usr/bin/env python3
"""
Script to run maximum tolerable perturbation analysis.

This script calculates the maximum tolerable perturbation (basin of attraction) 
for each gait in the gait space and generates plots comparing the results 
with and without feedback control.
"""


def main():
    """Main function to run the maximum tolerable perturbation analysis."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run maximum tolerable perturbation analysis'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run analysis on a small subset of gaits for testing'
    )
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='Only plot from saved results, do not run analysis'
    )
    
    args = parser.parse_args()
    
    # Default file paths
    solutions_file = 'data/simplest_walker/limit_cycle_solutions.npz'
    linear_analysis_file = 'data/simplest_walker/linear_analysis_results.npz'
    output_file = 'data/simplest_walker/maximum_tolerable_perturbation_results.npz'
    plot_file = 'data/simplest_walker/maximum_tolerable_perturbation_plot.png'
    
    # Check if input files exist
    if not os.path.exists(solutions_file):
        print(f"Error: Solutions file not found: {solutions_file}")
        sys.exit(1)
    
    if not os.path.exists(linear_analysis_file):
        print(f"Error: Linear analysis file not found: {linear_analysis_file}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plot_dir = os.path.dirname(plot_file)
    if plot_dir and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    print("Starting Maximum Tolerable Perturbation Analysis")
    print("=" * 50)
    print(f"Solutions file: {solutions_file}")
    print(f"Linear analysis file: {linear_analysis_file}")
    print(f"Output file: {output_file}")
    print(f"Plot file: {plot_file}")
    print(f"Test mode: {args.test_mode}")
    print(f"Plot only: {args.plot_only}")
    print("=" * 50)
    
    try:
        # Create analysis object
        analysis = MaximumTolerablePerturbation(
            solutions_file=solutions_file,
            linear_analysis_file=linear_analysis_file
        )
        
        if args.plot_only:
            # Plot-only mode: load existing results and plot
            print("\nPLOT-ONLY MODE - loading existing results...")
            
            if not os.path.exists(output_file):
                print(f"Error: Results file not found: {output_file}")
                print("Please run the analysis first (without --plot-only flag)")
                sys.exit(1)
            
            analysis.load_results(output_file)
            print("\nGenerating plots...")
            analysis.plot_results(save_path=plot_file)
            print(f"Plot saved to: {plot_file}")
            return
        
        if args.test_mode:
            # Test mode: run on a small subset of gaits
            print("\nRunning in TEST MODE - analyzing subset of gaits...")
            
            # Test a small subset (every 10th gait)
            test_indices = [(i, j) for i in range(0, 101, 10) for j in range(0, 101, 10)]
            
            # Initialize results matrices
            n_sl = len(analysis.target_step_lengths)
            n_sf = len(analysis.target_step_frequencies)
            analysis.max_tolerable_perturbation_with_control = np.full((n_sl, n_sf), np.nan)
            analysis.max_tolerable_perturbation_without_control = np.full((n_sl, n_sf), np.nan)
            
            # Run analysis for test subset
            print("Testing WITH feedback control...")
            for sl_idx, sf_idx in test_indices:
                result = analysis.calculate_max_tolerable_perturbation_for_gait(
                    sl_idx, sf_idx, with_control=True
                )
                analysis.max_tolerable_perturbation_with_control[sl_idx, sf_idx] = result
                print(f"Gait ({sl_idx}, {sf_idx}) with control: {result:.3f}")
            
            print("Testing WITHOUT feedback control...")
            for sl_idx, sf_idx in test_indices:
                result = analysis.calculate_max_tolerable_perturbation_for_gait(
                    sl_idx, sf_idx, with_control=False
                )
                analysis.max_tolerable_perturbation_without_control[sl_idx, sf_idx] = result
                print(f"Gait ({sl_idx}, {sf_idx}) without control: {result:.3f}")
        else:
            # Full analysis: run for all gaits
            print("\nRunning FULL ANALYSIS - analyzing all gaits in the grid...")
            analysis.run_analysis()
        
        # Save results
        analysis.save_results(output_file)
        
        # Plot results
        print("\nGenerating plots...")
        analysis.plot_results(save_path=plot_file)
        
        print("\nAnalysis completed successfully!")
        print(f"Results saved to: {output_file}")
        print(f"Plot saved to: {plot_file}")
        
        if args.test_mode:
            print("\nNote: Test mode was used. Run without --test-mode for full analysis.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 