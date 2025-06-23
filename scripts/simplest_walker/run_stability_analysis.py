import numpy as np
import matplotlib.pyplot as plt
from models.simplest_walker.analysis.utils import load_limit_cycle_solutions
from models.simplest_walker.analysis.analyze_gait_stability import analyze_gait_stability

def plot_stability_results(target_step_frequency, target_step_length, stability_metric):
    """Plot the stability analysis results."""
    plt.figure(figsize=(5, 5))
    plt.contour(target_step_frequency, target_step_length, stability_metric, 
                levels=np.linspace(0, 2, 1000))
    plt.colorbar(label='Max Eigenvalue Magnitude')
    plt.xlabel('Step Frequency (Hz)')
    plt.ylabel('Step Length (m)')
    plt.title('Stability Metric')
    plt.tight_layout()
    plt.show()

def main():
    """Main entry point for stability analysis for all limit cycle solutions."""
    # Load the limit cycle solutions and target ranges
    solutions, target_step_length, target_step_frequency = load_limit_cycle_solutions()

    # Perform stability analysis
    stability_metric, execution_time = analyze_gait_stability(solutions)
    
    print(f"Time taken: {execution_time} seconds")

    # Plot results
    plot_stability_results(target_step_frequency, target_step_length, stability_metric)

if __name__ == "__main__":
    main() 