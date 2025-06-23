import argparse
from models.simplest_walker.analysis.simulate_walker import simulate_walker, plot_simulation_results

def parse_args():
    parser = argparse.ArgumentParser(description='Run a bipedal walker simulation')
    parser.add_argument('--step-length', type=float, default=0.5,
                      help='Length of each step (default: 0.5)')
    parser.add_argument('--step-frequency', type=float, default=0.5,
                      help='Frequency of steps (default: 0.5)')
    parser.add_argument('--nn-steps', type=int, default=10,
                      help='Number of steps to simulate (default: 10)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    # Run simulation
    results = simulate_walker(args.step_length, args.step_frequency, args.nn_steps)

    # Plot results
    plot_simulation_results(results, args.step_length, args.step_frequency)

if __name__ == "__main__":
    main()