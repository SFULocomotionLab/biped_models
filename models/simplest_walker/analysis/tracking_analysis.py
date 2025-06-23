"""
Tracking analysis module for the simplest walker model.

This module provides functionality to analyze and visualize the tracking performance
of a bipedal walker model following different paths.
"""

import numpy as np
from numpy import flipud, flip
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import torch
from scipy.interpolate import interpn
from models.simplest_walker.SimplestWalker import SimplestWalker
from models.simplest_walker.analysis.utils import (load_limit_cycle_solutions, load_linear_analysis_data,
                   extract_gain_matrices, extract_matrices_from_solutions,
                   walker_state_interpol, feedback_gain_interpol)


class TrackingAnalysis:
    """Class for analyzing tracking performance of the simplest walker model."""

    def __init__(self, data_dir: str = 'data/simplest_walker'):
        """
        Initialize the tracking analysis.

        Args:
            data_dir: Directory containing the required data files
        """
        # Load required data
        self.solutions, self.target_step_lengths, self.target_step_frequencies = \
            load_limit_cycle_solutions()

        lad = load_linear_analysis_data()
        kppName = lad.files[0]
        kpp = lad[kppName]
        
        # Extract feedback gains
        self.K11, self.K12, self.K21, self.K22, self.K31, self.K32 = extract_gain_matrices(kpp)
        
        # Initialize walker with default parameters
        # Note: These will be updated when tracking starts
        x0 = np.array([0.0, 0.0])  # Initial state [stance_angle, stance_vel]
        s_nominal = np.array([0.35, 0.35])  # Nominal step length and frequency
        u_nominal = np.array([0.0, 0.0, 0.0])  # Nominal control inputs
        self.walker = SimplestWalker(x0, s_nominal, u_nominal)
        self.animation_enabled = False

    def generate_sinusoid_path(self, start_point: Tuple[float, float],
                             sine_params: Dict) -> Tuple[np.ndarray, List]:
        """
        Generate a sinusoidal path for tracking.

        Args:
            start_point: Starting point (SL, SF)
            sine_params: Dictionary containing sine wave parameters
                - ampSL: Amplitude for step length
                - ampSF: Amplitude for step frequency
                - freqSL: Frequency for step length (Hz)
                - freqSF: Frequency for step frequency (Hz)
                - time: Total time
                - n_steps: Number of steps

        Returns:
            Tuple containing:
            - path: Array of path points
            - path_ks_all: List of feedback gains for each point
        """
        t = np.linspace(0, sine_params['time'], sine_params['n_steps'])
        
        # Generate sinusoidal variations
        sl_variation = sine_params['ampSL'] * np.sin(2 * np.pi * sine_params['freqSL'] * t)
        sf_variation = sine_params['ampSF'] * np.sin(2 * np.pi * sine_params['freqSF'] * t)
        
        # Create path points
        path = np.zeros((sine_params['n_steps'], 7))
        path_ks_all = []
        
        for i in range(sine_params['n_steps']):
            current_sl = start_point[0] + sl_variation[i]
            current_sf = start_point[1] + sf_variation[i]
            
            # Get nominal state and control
            s_nominal, u_nominal = walker_state_interpol(current_sl, current_sf, self.solutions,
                                                       self.target_step_lengths,
                                                       self.target_step_frequencies)

            # Get feedback gains
            K = feedback_gain_interpol(current_sl, current_sf, self.K11, self.K12, self.K21,
                                     self.K22, self.K31, self.K32, self.target_step_lengths,
                                     self.target_step_frequencies)
            
            path[i, :] = np.concatenate([s_nominal[:2], u_nominal, np.array([current_sl, current_sf])])
            path_ks_all.append(K)
            
        return path, path_ks_all

    def generate_step_response_path(self, target_point: Tuple[float, float], 
                                  MPT: Tuple[float, float], perturbation_target: str,
                                  perturbation_direction: str,
                                  n_steps: int = 20) -> Tuple[np.ndarray, List]:
        """
        Generate a step response path for tracking.

        Args:
            target_point: Target point (SL, SF) to reach
            MPT(maximum tolerable perturbation): perturbation from target point (SL, SF)
            perturbation_target: perturbation assigned to SL or SF
            perturbation_direction: direction of perturbation (positive or negative)
            n_steps: Number of steps in the path (default: 20)

        Returns:
            Tuple containing:
            - path: Array of path points
            - path_ks_all: List of feedback gains for each point
        """
        # Create path points
        path = np.zeros((n_steps, 7))
        path_ks_all = []
        
        # Calculate start point by adding perturbation to target
        if perturbation_target == 'SL':
            if perturbation_direction == 'positive':
                start_point = (target_point[0] + MPT, target_point[1])
            elif perturbation_direction == 'negative':
                start_point = (target_point[0] - MPT, target_point[1])
            else:
                raise ValueError(f"Invalid perturbation direction: {perturbation_direction}")
        elif perturbation_target == 'SF':
            if perturbation_direction == 'positive':
                start_point = (target_point[0], target_point[1] + MPT)
            elif perturbation_direction == 'negative':
                start_point = (target_point[0], target_point[1] - MPT)
            else:
                raise ValueError(f"Invalid perturbation direction: {perturbation_direction}")
        else:
            raise ValueError(f"Invalid perturbation target: {perturbation_target}")
        
        # Generate path points
        for i in range(n_steps):
            # For first step, use start point, otherwise use target point
            current_point = start_point if i == 0 else target_point
            
            # Get nominal state and control
            s_nominal, u_nominal = walker_state_interpol(current_point[0], current_point[1], 
                                                       self.solutions,
                                                       self.target_step_lengths,
                                                       self.target_step_frequencies)

            # Get feedback gains
            K = feedback_gain_interpol(current_point[0], current_point[1], 
                                     self.K11, self.K12, self.K21,
                                     self.K22, self.K31, self.K32, 
                                     self.target_step_lengths,
                                     self.target_step_frequencies)
            
            path[i, :] = np.concatenate([s_nominal[:2], u_nominal, np.array([current_point[0], current_point[1]])])
            path_ks_all.append(K)
            
        return path, path_ks_all

    def track_path(self, path: np.ndarray, path_ks_all: List,
                  neural_net: Optional[torch.nn.Module] = None,
                  RLtrained: bool = False) -> Dict:
        """
        Track the generated path using either feedback control or neural network.

        Args:
            path: Array of path points to track
            path_ks_all: List of feedback gains for each point
            neural_net: Optional neural network for control
            RLtrained: Whether the neural network is RL-trained

        Returns:
            Dictionary containing tracking results:
            - step_lengths: Array of achieved step lengths
            - step_times: Array of achieved step times
            - rewards: Array of rewards
            - control_inputs: Array of control inputs
            - rmse_sl: Root mean square error of step length
            - rmse_sf: Root mean square error of step frequency
        """
        n_steps = len(path)
        step_lengths = np.zeros(n_steps)
        step_times = np.zeros(n_steps)
        step_frequencies = np.zeros(n_steps)
        rewards = np.zeros(n_steps)
        control_inputs = np.zeros((3, n_steps))
        rmse_sl = np.zeros(n_steps)
        rmse_sf = np.zeros(n_steps)

        # Initialize walker with first point
        starting_point = path[0]
        x0 = starting_point[:2]
        s_nominal = starting_point[5:6]
        u_nominal = starting_point[2:5]
        self.walker = SimplestWalker(x0, s_nominal, u_nominal)
        
        u0 = starting_point[2:5]
        step_lengths[0] = starting_point[5]
        step_times[0] = 1/starting_point[6]
        sl = starting_point[5]
        st = 1/starting_point[6]

        for i in range(n_steps):
            target_point = path[i]

            if neural_net is not None:
                # Neural network control
                device = next(neural_net.parameters()).device  # Get the device of the neural network
                xs = torch.tensor([[target_point[5], target_point[6], sl, 1/st]], 
                                dtype=torch.float32, device=device)
                neural_net.eval()
                with torch.no_grad():
                    next_u = neural_net(xs)
                u0 = next_u.detach().cpu().numpy().flatten()

                if RLtrained:
                    # to remove the tanh effect from the policy (set by default in SB3)
                    # Clip actions to prevent numerical instability
                    u0 = np.clip(u0, -0.999, 0.999) # to avoid division by zero below
                    u0 = np.arctanh(u0)  # More stable than log-based formula
            else:
                # Feedback control
                self.walker.s_nominal = target_point[:2]
                self.walker.u_nominal = target_point[2:5]
                self.walker.K = path_ks_all[i]
                u0 = self.walker.apply_feedback_controller(x0, self.walker.K)

            # Take one step
            next_s, _, _ = self.walker.take_one_step(x0, u0)
            
            if self.walker.fall_flag:
                print("Walker fell down!")
                # sl = 1000
                # st = 0.001
                break

            # Calculate step measures
            sl, _, _, st = self.walker.get_step_measures(next_s)
            step_lengths[i] = sl
            step_times[i] = st
            
            # Update state
            x0 = next_s[:2]
            
            # Calculate reward
            rmse_sl[i] = np.sqrt((sl - target_point[5])**2)
            rmse_sf[i] = np.sqrt((1/st - target_point[6])**2)
            rewards[i] = 1 - (rmse_sl[i] + 10 * rmse_sf[i])
            
            control_inputs[:, i] = u0

        return {
            'step_lengths': step_lengths,
            'step_times': step_times,
            'rewards': rewards,
            'control_inputs': control_inputs,
            'rmse_sl': rmse_sl,
            'rmse_sf': rmse_sf
        }

    def plot_results(self, path: np.ndarray, results: Dict, title: str = ''):
        """
        Plot tracking results.

        Args:
            path: Original path points
            results: Dictionary containing tracking results
            title: Title for the plot
        """
        # Set up color scheme
        line_col1 = np.array([0, 83, 155]) / 255
        line_col2 = np.array([204, 6, 51]) / 255

        # Create a single figure with subplots
        fig = plt.figure(figsize=(12, 5))
        fig.suptitle(title)
        
        # First subplot for path tracking
        ax1 = plt.subplot(121)
        ax1.plot(path[:, 6], path[:, 5], '-', color=line_col1, linewidth=2,
                label='Desired')
        ax1.plot(1/results['step_times'], results['step_lengths'], '-o',
                color=line_col2, linewidth=2, label='Tracked')
        
        ax1.set_xlabel('Step frequency [dimensionless]')
        ax1.set_ylabel('Step length [dimensionless]')
        ax1.set_xlim([0.1, 1.1])
        ax1.set_ylim([0.11, 1.1])
        ax1.set_xticks([0.1, 0.6, 1.1])
        ax1.set_yticks([0.1, 0.6, 1.1])
        ax1.legend()
        ax1.grid(True)
        
        # Second subplot for time series
        ax2 = plt.subplot(122)
        ax2.plot(path[:, 5], '-', color=line_col1, linewidth=2, label='Desired SL')
        ax2.plot(results['step_lengths'], '-o', color=line_col2, linewidth=2, label='Tracked SL')
        ax2.plot(path[:, 6], '-', color=line_col1, linewidth=2, label='Desired SF', alpha=0.5)
        ax2.plot(1/results['step_times'], '-o', color=line_col2, linewidth=2, label='Tracked SF', alpha=0.5)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Step length/frequency [dimensionless]')
        ax2.set_ylim([0.1, 1.1])
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        
        # Save the plot instead of showing it
        import os
        os.makedirs('logs/plots', exist_ok=True)
        plt.savefig(f'logs/plots/{title.replace(" ", "_")}.png')
        plt.close()  # Close the figure to free memory
