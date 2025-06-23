"""
Utility functions for the simplest walker model analysis.

This module provides common utility functions used across different analysis modules.
"""

import numpy as np
from scipy.interpolate import interpn
from typing import Dict, List, Tuple, Optional
import concurrent.futures
from functools import partial


def load_limit_cycle_solutions(file_path='data/simplest_walker/limit_cycle_solutions.npz'):
    """
    Load limit cycle solutions from a .npz file.
    
    Args:
        file_path (str): Path to the .npz file containing limit cycle solutions
        
    Returns:
        tuple: (solutions, target_step_length, target_step_frequency)
    """
    try:
        loaded_data = np.load(file_path, allow_pickle=True)
        solutions = loaded_data['arr_0']
        
        # Define the target ranges
        target_step_length = np.linspace(1.1, 0.1, 101)
        target_step_frequency = np.linspace(0.1, 1.1, 101)
        
        return solutions, target_step_length, target_step_frequency
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find limit cycle solutions file at {file_path}")
    except KeyError:
        raise KeyError("The .npz file does not contain the expected 'arr_0' key")
    except Exception as e:
        raise Exception(f"Error loading limit cycle solutions: {str(e)}")


def load_linear_analysis_data(file_path='data/simplest_walker/linear_analysis_results.npz'):
    """
    Load data from linear analysis results.
    
    Args:
        file_path (str): Path to the .npz file containing linear analysis results
        
    Returns:
        dict: Loaded linear analysis data
    """
    try:
        return np.load(file_path, allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find linear analysis results file at {file_path}")
    except Exception as e:
        raise Exception(f"Error loading linear analysis results: {str(e)}")


def extract_gain_matrices(K):
    """
    Extract feedback gain matrices from kpp array.
        
    Args:
        K: Array containing feedback gains across step length and frequency ranges
            
    Returns:
        Tuple containing 6 gain matrices (K11, K12, K21, K22, K31, K32),
        each 101x101 representing gains across step length and frequency ranges
    """
    # Initialize gain matrices
    K11 = np.zeros((101, 101))
    K12 = np.zeros((101, 101)) 
    K21 = np.zeros((101, 101))
    K22 = np.zeros((101, 101))
    K31 = np.zeros((101, 101))
    K32 = np.zeros((101, 101))

    # Fill matrices by extracting elements from kpp
    for i in range(101):
        for j in range(101):
            K11[i,j] = K[i][j][0,0]
            K12[i,j] = K[i][j][0,1]
            K21[i,j] = K[i][j][1,0]
            K22[i,j] = K[i][j][1,1]
            K31[i,j] = K[i][j][2,0]
            K32[i,j] = K[i][j][2,1]
                
    return K11, K12, K21, K22, K31, K32


def extract_matrices_from_solutions(solutions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract matrices from solutions.
    
    Args:
        solutions: Array containing solutions from optimization
        
    Returns:
        Tuple containing:
        - all_stance_vel: Matrix of stance velocities
        - all_swing_vel: Matrix of swing velocities
        - all_pushoff: Matrix of pushoff values
        - all_hip_stiffness: Matrix of hip stiffness values
    """
    solutions_f = solutions[0]

    # Extract stance velocities (column 2)
    all_stance_vel = np.array([x[:, 1] for x in solutions_f])
    
    # Extract swing velocities (column 4)
    all_swing_vel = np.array([x[:, 3] for x in solutions_f])
    
    # Extract pushoff values (column 3)
    all_pushoff = np.array([x[:, 2] for x in solutions_f])
    
    # Extract hip stiffness values (column 4)
    all_hip_stiffness = np.array([x[:, 3] for x in solutions_f])
    
    return all_stance_vel, all_swing_vel, all_pushoff, all_hip_stiffness


def walker_state_interpol(sl: float, sf: float, solutions: np.ndarray, 
                         target_step_lengths: np.ndarray, 
                         target_step_frequencies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate walker state for given step length and frequency using limit cycle solutions.
    
    Args:
        sl: Step length
        sf: Step frequency
        solutions: Array of limit cycle solutions
        target_step_lengths: Array of target step lengths
        target_step_frequencies: Array of target step frequencies
        
    Returns:
        Tuple containing:
        - new_q: New state vector [stance_angle, stance_velocity, swing_angle, swing_velocity]
        - new_u: New control vector [pushoff, hip_stiffness]
    """
    # Extract matrices from solutions
    all_stance_vel, all_swing_vel, all_pushoff, all_hip_stiffness = \
        extract_matrices_from_solutions(solutions)
    
    # Calculate stance and swing angles from geometry
    new_stance_ang = np.arcsin(sl/2)
    new_swing_ang = 2 * new_stance_ang
    
    # Create points for interpolation
    points = (target_step_lengths, target_step_frequencies)
    
    # Prepare values for interpolation
    stance_vel_values = all_stance_vel.T
    swing_vel_values = all_swing_vel.T  
    pushoff_values = all_pushoff.T
    hip_stiffness_values = all_hip_stiffness.T
    
    # Interpolate each component
    new_stance_vel = interpn(points, stance_vel_values, np.array([[sl, sf]]), method='linear')[0]
    new_swing_vel = interpn(points, swing_vel_values, np.array([[sl, sf]]), method='linear')[0]
    new_pushoff = interpn(points, pushoff_values, np.array([[sl, sf]]), method='linear')[0]
    new_hip_stiffness = interpn(points, hip_stiffness_values, np.array([[sl, sf]]), method='linear')[0]
    
    # Combine into state and control vectors
    new_q = np.array([new_stance_ang, new_stance_vel, new_swing_ang, new_swing_vel])
    new_u = np.array([new_pushoff, new_hip_stiffness, new_hip_stiffness])
    
    return new_q, new_u


def feedback_gain_interpol(sl: float, sf: float, K11: np.ndarray, K12: np.ndarray,
                         K21: np.ndarray, K22: np.ndarray, K31: np.ndarray, K32: np.ndarray,
                         target_step_lengths: np.ndarray, 
                         target_step_frequencies: np.ndarray) -> np.ndarray:
    """
    Interpolate feedback gains for given step length and frequency using linear analysis results.
    
    Args:
        sl: Step length
        sf: Step frequency
        K11-K32: Feedback gain matrices
        target_step_lengths: Array of target step lengths
        target_step_frequencies: Array of target step frequencies
        
    Returns:
        Interpolated feedback gain matrix K
    """
    # Create points for interpolation
    points = (target_step_lengths, target_step_frequencies)
    
    # Interpolate each gain component
    new_K11 = interpn(points, K11, np.array([[sl, sf]]), method='linear')[0]
    new_K12 = interpn(points, K12, np.array([[sl, sf]]), method='linear')[0]
    new_K21 = interpn(points, K21, np.array([[sl, sf]]), method='linear')[0]
    new_K22 = interpn(points, K22, np.array([[sl, sf]]), method='linear')[0]
    new_K31 = interpn(points, K31, np.array([[sl, sf]]), method='linear')[0]
    new_K32 = interpn(points, K32, np.array([[sl, sf]]), method='linear')[0]
    
    # Combine into feedback gain matrix
    new_K = np.array([[new_K11, new_K12],
                     [new_K21, new_K22],
                     [new_K31, new_K32]])
    
    return new_K 


def control_function(solutions: np.ndarray, step_lengths: np.ndarray, 
                    step_frequencies: np.ndarray, all_K: np.ndarray,
                    current_slsf: np.ndarray, target_slsf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate control inputs required to reach target step length and frequency.
    
    This function takes in current state of the simplest walker and outputs the controls
    (pushoff and stiffnesses) required for the target point using interpolation functions.
    
    Note: This function assumes "existence" of such controls. In other words, it assumes
    the walker can reach the target state without falling down. Therefore, the target point
    should be "close enough" to the current point.
    
    Args:
        solutions: Array of limit cycle solutions
        step_lengths: Array of target step lengths
        step_frequencies: Array of target step frequencies
        all_K: Array of feedback gain matrices
        current_slsf: Current [step_length, step_frequency]
        target_slsf: Target [step_length, step_frequency]
        
    Returns:
        Tuple containing:
        - u: Control inputs [pushoff, stiffness1, stiffness2]
        - current_state: Current state [theta, theta_dot]
        - q_target: Target state [theta, theta_dot]
        - u_target: Target control inputs
    """
    # Get current states from current SL SF
    current_state, current_control = walker_state_interpol(
        current_slsf[0], current_slsf[1], 
        solutions, step_lengths, step_frequencies)
    current_state = current_state[:2]  # Take only theta and theta_dot
    
    # Calculate limit-cycle values of target point
    q_target, u_target = walker_state_interpol(
        target_slsf[0], target_slsf[1],
        solutions, step_lengths, step_frequencies)
    q_target = q_target[:2]  # Take only theta and theta_dot
    
    # Get feedback gains for target point
    K_target = feedback_gain_interpol(
        target_slsf[0], target_slsf[1],
        *all_K, step_lengths, step_frequencies)

    # Calculate control input using feedback law
    state_error = current_state - q_target

    u = u_target - K_target @ state_error
    
    return u, current_state, q_target, u_target


def generate_nn_data(solutions: np.ndarray, target_step_lengths: np.ndarray, 
                    target_step_frequencies: np.ndarray, all_K: np.ndarray,
                    n_samples: int, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data for neural network feedback controller.
    
    Args:
        solutions: Array of limit cycle solutions
        target_step_lengths: Array of target step lengths
        target_step_frequencies: Array of target step frequencies  
        all_K: Array of feedback gain matrices
        n_samples: Number of random samples to generate
        sigma: Standard deviation for perturbations
        
    Returns:
        Tuple containing:
            - X: Array of input data [target_sl, target_sf, current_sl, current_sf]
            - Y: Array of output data (control inputs)
    """
    # Define bounds for random sampling
    lower_bound = np.min(target_step_lengths) + sigma   
    upper_bound = np.max(target_step_lengths) - sigma
    
    # Generate random target points
    random_tsl = np.random.uniform(lower_bound, upper_bound, n_samples)
    random_tsf = np.random.uniform(lower_bound, upper_bound, n_samples)
    
    # Generate random current points as perturbations from targets
    random_csl = random_tsl + sigma * (2 * np.random.rand(n_samples) - 1)
    random_csf = random_tsf + sigma * (2 * np.random.rand(n_samples) - 1)
    
    # Initialize output arrays
    Y = np.zeros((n_samples, 3))
    X = np.zeros((n_samples, 4))

    # Get control inputs for each sample using control_function
    valid_indices = []
    for i in range(n_samples):
        current_slsf = np.array([random_csl[i], random_csf[i]])  # current SL, SF
        target_slsf = np.array([random_tsl[i], random_tsf[i]])   # target SL, SF
        
        u, _, current_state, _ = control_function(
            solutions, target_step_lengths, target_step_frequencies,
            all_K, current_slsf, target_slsf)
        
        # Only keep samples where pushoff (first element of u) is non-negative
        if u[0] >= 0:
            valid_indices.append(i)
            Y[i] = u
            X[i] = np.array([random_tsl[i], random_tsf[i], current_slsf[0], current_slsf[1]])
    
    # Filter arrays to keep only valid entries
    X = X[valid_indices]
    Y = Y[valid_indices]
    
    return X, Y