#!/usr/bin/env python3
import argparse
import numpy as np
from models.simplest_walker.optimization.find_gait import find_limit_cycle

def main():
    parser = argparse.ArgumentParser(description='Find limit cycle solution for the simplest walker')
    
    # Add arguments
    parser.add_argument('--init-guess', type=float, nargs=5, default=[0.25, -0.25, 0.1, .2, .2],
                      help='Initial guess vector [stance_angle, stance_vel, pushoff, spring_const1, spring_const2]')
    parser.add_argument('--target-step-length', type=float, default=0.5,
                      help='Target step length')
    parser.add_argument('--target-frequency', type=float, default=0.5,
                      help='Target step frequency (steps per second)')
    parser.add_argument('--method', type=str, default='trust-constr',
                      choices=['SLSQP', 'trust-constr'],
                      help='Optimization method to use')
    
    args = parser.parse_args()
    
    # Convert initial guess to numpy array
    init_guess = np.array(args.init_guess)
    
    print("\nStarting optimization with parameters:")
    print(f"Initial guess: {init_guess}")
    print(f"Target step length: {args.target_step_length}")
    print(f"Target frequency: {args.target_frequency}")
    print(f"Optimization method: {args.method}\n")
    
    # Find limit cycle
    result = find_limit_cycle(
        init_guess=init_guess,
        method=args.method,
        target_step_length=args.target_step_length,
        target_frequency=args.target_frequency
    )
    
    if result:
        solution, success, message = result
        print("\nOptimization Results:")
        print(f"Success: {success}")
        print(f"Message: {message}")
        print("\nSolution:")
        print(f"Stance angle: {solution[0]:.4f} rad")
        print(f"Stance velocity: {solution[1]:.4f}")
        print(f"Pushoff: {solution[2]:.4f}")
        print(f"Spring constant: {solution[3]:.4f}")
    else:
        print("\nOptimization failed to find a solution")

if __name__ == "__main__":
    main() 