import numpy as np
from models.simplest_walker.SimplestWalker import SimplestWalker
from control import place, dlqr
import time
from .utils import load_limit_cycle_solutions

    
def perform_linear_analysis(solutions, target_step_length, target_step_frequency):
    """Perform linear analysis and calculate feedback gains for all gaits.
    
    Args:
        solutions: Array of limit cycle solutions
        target_step_length: Array of target step lengths
        target_step_frequency: Array of target step frequencies
        
    Returns:
        dict: Dictionary containing all analysis results
    """
    # Initialize storage arrays
    K_pp = np.zeros((101, 101, 3, 2))  # Pole placement gains
    K_lqr = np.zeros((101, 101, 3, 2))  # LQR gains
    A_matrices = np.zeros((101, 101, 2, 2))
    B_matrices = np.zeros((101, 101, 2, 3)) 
    C_matrices = np.zeros((101, 101, 2, 2))
    D_matrices = np.zeros((101, 101, 2, 3))
    eig_real_matrices = np.zeros((101, 101, 2))
    eig_imag_matrices = np.zeros((101, 101, 2))

    start_time = time.time()
    # Loop through all gaits
    for sf in range(101):
        print(f"Processing gait SF={target_step_frequency[sf]}")
        for sl in range(101):
            # Extract solution
            solution = solutions[0][sf][sl,:]
            x0 = solution[:2]
            u_nominal = solution[2:4]
            s_nominal = x0.copy()
            
            # Create control vector
            u0 = np.zeros(3)
            u0[0] = u_nominal[0]  # pushoff
            u0[1] = u_nominal[1]  # spring constant 1
            u0[2] = u0[1]         # spring constant 2

            # Initialize walker
            obj = SimplestWalker(s_nominal, s_nominal, u_nominal)
            obj.anim = False

            # Get linear dynamics matrices
            A, B, C, D = obj.do_linear_analysis(x0, u0)
            # Calculate eigenvalues of A matrix
            eig_vals = np.linalg.eigvals(A)
            eig_vals_real = np.real(eig_vals)
            eig_vals_imag = np.imag(eig_vals)

            # Store matrices
            A_matrices[sl,sf] = A
            B_matrices[sl,sf] = B
            C_matrices[sl,sf] = C 
            D_matrices[sl,sf] = D
            eig_real_matrices[sl,sf] = eig_vals_real
            eig_imag_matrices[sl,sf] = eig_vals_imag

            try:
                # Pole placement
                poles = np.array([0.0, 0.0])  # Place poles at origin
                K_pp[sl,sf] = place(A, B, poles)

                # LQR design
                Q = np.eye(2)  # State cost
                R = 0.001 * np.eye(3)  # Control cost
                K_lqr[sl,sf] = dlqr(A, B, Q, R)[0]
                
            except:
                print(f"Control design failed for SL={target_step_length[sl]}, SF={target_step_frequency[sf]}")
                continue

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    return {
        'K_pp': K_pp,
        'K_lqr': K_lqr,
        'A': A_matrices,
        'B': B_matrices,
        'C': C_matrices,
        'D': D_matrices,
        'eig_real': eig_real_matrices,
        'eig_imag': eig_imag_matrices
    } 