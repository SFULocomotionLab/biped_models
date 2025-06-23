import numpy as np
import matplotlib.pyplot as plt
from models.simplest_walker.SimplestWalker import SimplestWalker
from models.simplest_walker.analysis.utils import load_limit_cycle_solutions


def simulate_walker(step_length=0.5, step_frequency=0.5, nn_steps=10):
    # Load the limit cycle solutions
    solutions, target_step_length, target_step_frequency = load_limit_cycle_solutions()

    # Find the index of the target step length and step frequency
    trr = np.argmin(np.abs(target_step_length - step_length))
    tcc = np.argmin(np.abs(target_step_frequency - step_frequency))

    # Pick a specific gait
    solution = solutions[0][tcc][trr,:]

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

    # Initialize storage for trajectories
    trajn_all = None
    timen_all = []

    # Simulation parameters
    next_state = np.zeros((4, nn_steps))
    steplength = np.zeros(nn_steps)
    speed = np.zeros(nn_steps)
    steptime = np.zeros(nn_steps)
    stepfreq = np.zeros(nn_steps)
    W_toe = np.zeros(nn_steps)
    W_swing = np.zeros(nn_steps)
    W_total = np.zeros(nn_steps)

    # Run simulation
    for i in range(nn_steps):
        # Take one step
        next_state[:,i], trajn, timen = obj.take_one_step(x0, u0)
        
        # Calculate step measures
        steplength[i], speed[i], hip_dist, steptime[i] = obj.get_step_measures(next_state[:,i])
        stepfreq[i] = 1/steptime[i]

        # Calculate energy
        W_toe[i], W_swing[i], W_total[i] = obj.calculate_energy()
        
        # Update initial state for next step
        x0 = next_state[0:2,i]
        
        # Concatenate trajectories
        if trajn_all is None:
            trajn_all = trajn
            timen_all = timen
        else:
            trajn_all = np.hstack((trajn_all, trajn))
            end_time = timen_all[-1]
            timen_all = np.hstack((timen_all, timen + end_time))

        # Check for falls or no foot contact
        if obj.fall_flag:
            print("Walker fell down!")
            break
        elif not obj.foot_contact_flag:
            print("No foot contact detected!")
            break

    # Return simulation results
    return {
        'time': timen_all,
        'trajectories': trajn_all,
        'step_length': steplength,
        'step_frequency': stepfreq,
        'energy': {
            'toe': W_toe,
            'swing': W_swing,
            'total': W_total
        },
        'fall_flag': obj.fall_flag,
        'foot_contact_flag': obj.foot_contact_flag
    }


def plot_simulation_results(results, step_length, step_frequency):
    """Plot the simulation results."""
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(results['time'], results['trajectories'][0,:], label='Stance Angle')
    plt.plot(results['time'], results['trajectories'][2,:], label='Swing Angle')
    plt.plot(results['time'], results['trajectories'][2,:]-results['trajectories'][0,:], label='Inter-leg Angle')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title(f'Walker Joint Angles for step length = {step_length} and step frequency = {step_frequency}')

    plt.subplot(2, 1, 2)
    plt.plot(results['step_length'], label='Step Length')
    plt.plot(results['step_frequency'], label='Step Frequency')
    plt.xlim(0, len(results['step_length']))
    plt.ylim(0, 2)
    plt.legend()
    plt.grid(True)
    plt.xlabel('Step Number')
    plt.ylabel('Length/Frequency')
    plt.title('Step Measures')

    plt.tight_layout()
    plt.show()
