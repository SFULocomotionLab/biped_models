import numpy as np
import time
from models.simplest_walker.SimplestWalker import SimplestWalker

def analyze_gait_stability(solutions):
    """
    Analyze stability of all gait solutions.
    
    Args:
        solutions: Array of gait solutions
        target_step_length: Array of target step lengths
        target_step_frequency: Array of target step frequencies
        
    Returns:
        stability_metric: 2D array of stability metrics
        execution_time: Time taken for analysis
    """
    # Initialize storage for stability metric
    stability_metric = np.zeros((101, 101))
    
    start_time = time.time()
    # Loop through all solutions
    for sf in range(0, 101, 1):
        print(f"Processing gait column {sf}")
        for sl in range(0, 101, 1):
            # Extract solution for this gait
            solution = solutions[0][sf][sl,:]
            
            # Extract initial conditions and control parameters
            x0 = solution[:2]  # Initial state [stance_angle, stance_velocity]
            u_nominal = solution[2:4]  # Control parameters [pushoff, spring_constant]
            s_nominal = x0.copy()  # Nominal state
            
            # Create control vector (adding duplicate spring constant)
            u0 = np.zeros(3)
            u0[0] = u_nominal[0]  # pushoff
            u0[1] = u_nominal[1]  # spring constant
            u0[2] = u0[1]        # duplicate spring constant

            # Initialize walker object
            obj = SimplestWalker(s_nominal, s_nominal, u_nominal)
            obj.anim = False

            # Calculate stability (Jacobian eigenvalues)
            eig_J, _ = obj.calculate_linear_stability(x0, u0, eps=1e-6)
            
            # Store maximum eigenvalue magnitude for stability metric
            stability_metric[sl, sf] = np.max(np.abs(eig_J))

    end_time = time.time()
    execution_time = end_time - start_time
    
    return stability_metric, execution_time 