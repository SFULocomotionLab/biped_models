import numpy as np
from models.simplest_walker.analysis.utils import load_limit_cycle_solutions
from models.simplest_walker.analysis.linear_analysis import perform_linear_analysis


def main():
    """Entry point for linear dynamics analysis.
    
    Loads limit cycle solutions and performs linear analysis around nominal trajectories
    for a grid of step lengths and frequencies. Calculates pole placement and LQR feedback gains.
    
    Saves:
    - Pole placement gains (K_pp)
    - LQR gains (K_lqr) 
    - Linear dynamics matrices (A, B, C, D)
    """
    # Load data using the data loader
    solutions, target_step_length, target_step_frequency = load_limit_cycle_solutions()
    
    # Perform analysis
    results = perform_linear_analysis(solutions, target_step_length, target_step_frequency)
    
    # Save results
    # np.savez('data/linear_analysis_results.npz', **results)


if __name__ == "__main__":
    main()
