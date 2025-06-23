import numpy as np
import argparse
from models.simplest_walker.plotting.plot_functions import (
    plot_eigenvalues,
    plot_gains,
    plot_AB,
    plot_CD,
    plot_states,
    plot_controls
)
from models.simplest_walker.analysis.utils import load_limit_cycle_solutions, load_linear_analysis_data

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot various analysis results for the simplest walker model')
    parser.add_argument('--plots', nargs='+', choices=['eigenvalues', 'gains', 'AB', 'CD', 'states', 'controls', 'all'],
                      default=['all'], help='Which plots to generate')
    args = parser.parse_args()
    
    # Load all required data
    linear_data = load_linear_analysis_data()
    limit_cycle_data, target_step_length, target_step_frequency = load_limit_cycle_solutions()
    
    # Determine which plots to generate
    plots_to_generate = args.plots
    if 'all' in plots_to_generate:
        plots_to_generate = ['eigenvalues', 'gains', 'AB', 'CD', 'states', 'controls']
    
    # Generate requested plots
    for plot_type in plots_to_generate:
        print(f"Generating {plot_type} plot...")
        if plot_type == 'eigenvalues':
            plot_eigenvalues(linear_data, target_step_length, target_step_frequency, plot=True)
        elif plot_type == 'gains':
            plot_gains(linear_data, target_step_length, target_step_frequency, plot=True)
        elif plot_type == 'AB':
            plot_AB(linear_data, target_step_length, target_step_frequency, plot=True)
        elif plot_type == 'CD':
            plot_CD(linear_data, target_step_length, target_step_frequency, plot=True)
        elif plot_type == 'states':
            plot_states(limit_cycle_data, target_step_length, target_step_frequency, plot=True)
        elif plot_type == 'controls':
            plot_controls(limit_cycle_data, target_step_length, target_step_frequency, plot=True)

if __name__ == "__main__":
    main() 