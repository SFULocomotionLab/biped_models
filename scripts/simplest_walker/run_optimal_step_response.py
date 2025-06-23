"""
Test script for the OptimalStepResponseAnalysis module.

This script demonstrates how to use the completed module to analyze and optimize
step response of the simplest walker model.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.simplest_walker.analysis.optimal_step_response import OptimalStepResponseAnalysis


def test_optimal_step_response():
    """Test the optimal step response analysis."""
    
    # Initialize the analysis
    print("Initializing OptimalStepResponseAnalysis...")
    analysis = OptimalStepResponseAnalysis()
    
    # Define target and start points
    target_point = (0.6, 0.6)  # Target (SL, SF)
    
    # Create starting points 15% above and below target step length
    sl_target = target_point[0]
    sf_target = target_point[1]
    
    start_points = np.array([
        [sl_target * 1.0, sf_target*0.85],  # 15% below target
        [sl_target * 1.0, sf_target*1.15]   # 15% above target
    ])  # Multiple start points (n x 2)
    
    print(f"Target point: SL={target_point[0]:.2f}, SF={target_point[1]:.2f}")
    print(f"Start points:")
    for i, start in enumerate(start_points):
        print(f"  Point {i+1}: SL={start[0]:.3f}, SF={start[1]:.2f}")
    
    n_steps = 10
    # Complete analysis
    print("\nRunning complete analysis...")
    results = analysis.analyze_optimal_step_response(target_point, start_points, n_steps=n_steps)
    
    print(f"Complete analysis results:")
    print(f"  Optimal tau: {results['optimization_results']['optimal_tau']:.4f}")
    print(f"  Final error: {results['optimization_results']['optimal_error']:.6f}")
    print(f"  Number of paths analyzed: {len(results['reference_paths'])}")
    
    # Plot results
    print("\nGenerating plots...")
    fig = analysis.plot_optimal_step_response(results, "Optimal Step Response - 15% Perturbation")
    
    print("\nTest completed successfully!")
    return results


if __name__ == "__main__":
    # Run test
    print("="*60)
    print("TESTING OPTIMAL STEP RESPONSE ANALYSIS")
    print("="*60)
    
    try:
        # Run the 15% perturbation test
        results = test_optimal_step_response()
        
        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc() 