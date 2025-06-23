import numpy as np
from scipy.optimize import minimize
from models.simplest_walker.SimplestWalker import SimplestWalker

def find_limit_cycle(init_guess, method, target_step_length=0.5, target_frequency=0.5):
    """
    Find the limit-cycle solution for the simplest walker.
    
    Args:
        init_guess: Initial guess for optimization variables [stance_angle, stance_vel, pushoff, spring_const]
        method: Optimization method to use
        target_step_length: Desired step length for the gait
        target_frequency: Desired step frequency (steps per second)
    
    Returns:
        Optimization result if successful, False otherwise
    """
    # Initial guess
    q0 = init_guess[:2]
    u0 = init_guess[2:]
   
    # Create walker instance
    walker = SimplestWalker(q0, q0, u0)
    
    # Optimization options
    opt_options = {
        'maxiter': 1000,
        'xtol': 1e-8,
        'gtol': 1e-8,
        'disp': True,
        'verbose': 2
    }
    
    # Set up optimization problem
    bounds = [
        (-np.pi/2, np.pi/2),    # stance angle
        (-10, 10),              # stance velocity  
        (0, 100),                 # pushoff
        (-100, 100),              # spring constant
        (-100, 100)               # spring constant
    ]
    
    constraints = [
        {
            'type': 'eq',
            'fun': lambda x: periodicity_constraint(x, walker),
        },
        {
            'type': 'eq',
            'fun': lambda x: step_length_constraint(x, walker, target_step_length),
        },
        {
            'type': 'eq',
            'fun': lambda x: frequency_constraint(x, walker, target_frequency),
        },
        {
            'type': 'eq',
            'fun': lambda x: hip_spring_constraint(x),
        },
        {
            'type': 'ineq',
            'fun': lambda x: pushoff_constraint(x),
        }
    ]
    
    # Run optimization
    result = minimize(
        objective_func,
        init_guess,
        method=method,
        bounds=bounds,
        constraints=constraints,
        options=opt_options,
    )
    
    if result.success:
        print("Optimization successful")
        return result.x, result.success, result.message
    else:
        print("Optimization failed to find a stable limit cycle")
        return False

def objective_func(x):
    """
    Objective function to minimize.
    Minimizes nothing as it is just finding a limit cycle.
    """
    return 0
    
def periodicity_constraint(x, walker):
    """Constraint to ensure periodicity of the gait."""
    # Split optimization variables
    q = x[:2]  # state variables
    u = x[2:]  # control inputs
    walker.x0 = q
    walker.st_foot, walker.sw_foot, walker.hip = \
        walker.get_trajectory(q)

    # Take one step with current state and control
    next_state, _, _ = walker.take_one_step(q, u)
    
    # Check if walker fell or didn't make contact
    if walker.fall_flag or not walker.foot_contact_flag:
        return 1e6  # Return large value to indicate constraint violation
        
    # Calculate difference between initial and final states
    state_diff = next_state[:2] - q
    
    return np.linalg.norm(state_diff)
    
def step_length_constraint(x, walker, target_step_length):
    """Constraint to achieve desired step length."""
    # Split optimization variables
    q = x[:2]  # state variables
    u = x[2:]  # control inputs
    walker.x0 = q
    walker.st_foot, walker.sw_foot, walker.hip = \
        walker.get_trajectory(q)

    # Take one step
    next_state, _, _ = walker.take_one_step(q, u)
    
    next_state = next_state[0:2].T
    # Get step length
    step_length, _, _, _ = walker.get_step_measures(next_state)
    
    # Return difference from target
    return step_length - target_step_length
    
def frequency_constraint(x, walker, target_frequency):
    """Constraint to achieve desired step frequency."""
    # Split optimization variables
    q = x[:2]  # state variables
    u = x[2:]  # control inputs
    walker.x0 = q
    # Initialize posture
    walker.st_foot, walker.sw_foot, walker.hip = \
        walker.get_trajectory(q)

    # Take one step
    next_state, _, _ = walker.take_one_step(q, u)
    
    # Get step time
    _, _, _, step_time = walker.get_step_measures(next_state)
    
    # Calculate actual frequency
    actual_frequency = 1.0 / step_time
    
    # Return difference from target
    return actual_frequency - target_frequency

def hip_spring_constraint(x):
    """Constraint to ensure hip spring elements are equal."""
    # Get hip spring value (third element of control inputs)
    hip_spring = x[3]
    hip_spring_2 = x[4]
    # Return difference
    return hip_spring - hip_spring_2
    
def pushoff_constraint(x):
    """Constraint to ensure pushoff is non-negative."""
    # Get pushoff value (first element of control inputs)
    pushoff = x[2]
    
    # Return pushoff value (should be >= 0)
    return pushoff
        