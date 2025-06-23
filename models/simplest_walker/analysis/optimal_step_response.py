"""
Optimal step response analysis module for the simplest walker model.

This module provides functionality to analyze and optimize the step response
of a bipedal walker model when perturbed from a target gait. It creates
exponential reference paths, tracks them using feedback control, and optimizes
the time constant to minimize tracking error.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize

from models.simplest_walker.SimplestWalker import SimplestWalker
from models.simplest_walker.analysis.utils import (load_limit_cycle_solutions, load_linear_analysis_data,
                   extract_gain_matrices,
                   walker_state_interpol, feedback_gain_interpol)
from models.simplest_walker.analysis.tracking_analysis import TrackingAnalysis


class OptimalStepResponseAnalysis:
    """Class for analyzing and optimizing step response of the simplest walker model."""

    def __init__(self, data_dir: str = 'data/simplest_walker'):
        """
        Initialize the optimal step response analysis.

        Args:
            data_dir: Directory containing the required data files
        """
        # Load required data
        self.solutions, self.target_step_lengths, self.target_step_frequencies = \
            load_limit_cycle_solutions()

        lad = load_linear_analysis_data()
        kpp_name = lad.files[0]
        kpp = lad[kpp_name]
        
        # Extract feedback gains
        self.K11, self.K12, self.K21, self.K22, self.K31, self.K32 = extract_gain_matrices(kpp)
        
        # Initialize tracking analysis for path tracking
        self.tracking_analysis = TrackingAnalysis(data_dir)

    def _exp_conv(self, tau: float, t0: float, start_point: float, 
                 target_point: float, n_steps: int) -> np.ndarray:
        """
        Create exponential convergence trajectory.
        
        Args:
            tau: Time constant
            t0: Time delay
            start_point: Starting value
            target_point: Target value
            n_steps: Number of steps
            
        Returns:
            Array of values following exponential convergence
        """
        t = np.arange(1, n_steps+1)
        Y = target_point + (start_point - target_point) * np.exp(-(t-t0)/tau)
        return Y

    def generate_exponential_reference_path(self, target_point: Tuple[float, float],
                                          start_points: np.ndarray,
                                          tau: float, t0: float = 0.0,
                                          n_steps: int = 20) -> Tuple[list, list]:
        """
        Generate exponential reference paths from multiple start points to target point.
        
        Args:
            target_point: Target (SL, SF) point
            start_points: Array of starting (SL, SF) points (n x 2)
            tau: Time constant for exponential convergence
            t0: Time delay (default: 1.0)
            n_steps: Number of steps in the path
            
        Returns:
            Tuple containing:
            - paths: List of path arrays (n paths, each n_steps x 7)
            - paths_ks_all: List of feedback gains for each path
        """
        n_paths = len(start_points)
        paths = []
        paths_ks_all = []
        
        for i in range(n_paths):
            start_point = start_points[i]
            
            # Generate exponential trajectories for both SL and SF
            sl_traj = self._exp_conv(tau, t0, start_point[0], target_point[0], n_steps)
            sf_traj = self._exp_conv(tau, t0, start_point[1], target_point[1], n_steps)
            
            # Create path points
            path = np.zeros((n_steps, 7))
            path_ks_all = []
            
            for j in range(n_steps):
                current_sl = sl_traj[j]
                current_sf = sf_traj[j]
                
                # Get nominal state and control
                s_nominal, u_nominal = walker_state_interpol(current_sl, current_sf, self.solutions,
                                                           self.target_step_lengths,
                                                           self.target_step_frequencies)

                # Get feedback gains
                K = feedback_gain_interpol(current_sl, current_sf, self.K11, self.K12, self.K21,
                                         self.K22, self.K31, self.K32, self.target_step_lengths,
                                         self.target_step_frequencies)
                
                path[j, :] = np.concatenate([s_nominal[:2], u_nominal, np.array([current_sl, current_sf])])
                path_ks_all.append(K)
            
            paths.append(path)
            paths_ks_all.append(path_ks_all)
            
        return paths, paths_ks_all

    def calculate_tracking_error(self, tau: float, target_point: Tuple[float, float],
                               start_points: np.ndarray, t0: float = 0.0,
                               n_steps: int = 20) -> float:
        """
        Generate exponential reference paths, track them, and calculate total mean squared error.
        
        Args:
            tau: Time constant for exponential reference
            target_point: Target (SL, SF) point
            start_points: Array of starting (SL, SF) points (n x 2)
            t0: Time delay
            n_steps: Number of steps
            
        Returns:
            Total mean squared error across all paths
        """
        try:
            # Generate reference paths for all starting points
            paths, paths_ks_all = self.generate_exponential_reference_path(
                target_point, start_points, tau, t0, n_steps)
            
            n_paths = len(paths)
            total_rmse_sl = 0.0
            total_rmse_sf = 0.0
            total_rmse = 0.0
            
            # Track each path and calculate error
            for i in range(n_paths):
                # Track the path using tracking analysis
                results = self.tracking_analysis.track_path(paths[i], paths_ks_all[i])
                
                # Calculate MSE between tracked and reference
                tracked_sl = results['step_lengths']
                tracked_sf = 1 / results['step_times']  # Convert step times to frequencies
                
                reference_sl = paths[i][:, 5]  # Step lengths from path
                reference_sf = paths[i][:, 6]  # Step frequencies from path
                
                # Calculate MSE for both SL and SF for this path
                rmse_sl = np.sqrt(np.sum((tracked_sl - reference_sl) ** 2)/n_steps)
                rmse_sf = np.sqrt(np.sum((tracked_sf - reference_sf) ** 2)/n_steps)
                
                # Add to total MSE
                total_rmse_sl += rmse_sl
                total_rmse_sf += rmse_sf

            if start_points[0, 0] == target_point[0] and start_points[0, 1] != target_point[1]:
                total_rmse = total_rmse_sf
            elif start_points[0, 0] != target_point[0] and start_points[0, 1] == target_point[1]:
                total_rmse = total_rmse_sl
            else:
                raise ValueError("Check the start points")
            
            # fig, ax = plt.subplots()
            # ax.plot(reference_sl, '-o', label='Reference SL')
            # ax.plot(tracked_sl, '-o', label='Tracked SL')
            # ax.set_xlabel('Step')
            # ax.set_ylabel('Step Length')
            # ax.legend()
            # plt.show()
            # breakpoint()
            # exit()
            # Return average MSE across all paths
            return total_rmse
            
        except Exception as e:
            # Return a large error if tracking fails
            print(f"Tracking failed for tau={tau}: {e}")
            return 1e6

    def optimize_time_constant(self, target_point: Tuple[float, float],
                             start_points: np.ndarray,
                             tau_initial: float = 1, t0: float = 0.0,
                             n_steps: int = 20) -> Dict:
        """
        Optimize the time constant of the exponential reference path.
        
        Args:
            target_point: Target (SL, SF) point
            start_points: Array of starting (SL, SF) points (n x 2)
            tau_initial: Initial guess for time constant
            t0: Time delay
            n_steps: Number of steps
            
        Returns:
            Dictionary containing optimization results:
            - optimal_tau: Optimal time constant
            - optimal_error: Minimum tracking error
            - optimization_success: Whether optimization converged
            - optimization_message: Optimization message
        """
        # Define objective function for optimization
        def objective(tau):
            return self.calculate_tracking_error(tau, target_point, start_points, t0, n_steps)
        
        # Set bounds for time constant (must be positive)
        bounds = [(0.1, 10.0)]  # tau between 0.1 and 10.0
        
        # Optimize using scipy.optimize.minimize (similar to fminsearch in Matlab)
        result = minimize(
            objective,
            x0=[tau_initial],
            bounds=bounds,
            method='Nelder-Mead',  # Similar to fminsearch
            options={'maxiter': 100, 'disp': True}
        )
        
        return {
            'optimal_tau': result.x[0],
            'optimal_error': result.fun,
            'optimization_success': result.success,
            'optimization_message': result.message
        }

    def analyze_optimal_step_response(self, target_point: Tuple[float, float],
                                    start_points: np.ndarray,
                                    n_steps: int = 20) -> Dict:
        """
        Complete analysis: optimize time constant and generate results.
        
        Args:
            target_point: Target (SL, SF) point
            start_points: Array of starting (SL, SF) points (n x 2)
            n_steps: Number of steps
            
        Returns:
            Dictionary containing complete analysis results
        """
        # Optimize time constant
        opt_results = self.optimize_time_constant(target_point, start_points, n_steps=n_steps)
        
        # Generate optimal reference paths
        paths, paths_ks_all = self.generate_exponential_reference_path(
            target_point, start_points, opt_results['optimal_tau'], n_steps=n_steps)
        
        # Track optimal paths
        tracking_results = []
        for i in range(len(paths)):
            tracking_results.append(self.tracking_analysis.track_path(paths[i], paths_ks_all[i]))
        
        return {
            'optimization_results': opt_results,
            'reference_paths': paths,
            'tracking_results': tracking_results,
            'target_point': target_point,
            'start_points': start_points
        }

    def plot_optimal_step_response(self, results: Dict, title: str = 'Optimal Step Response'):
        """
        Plot the optimal step response results.
        
        Args:
            results: Dictionary containing analysis results
            title: Title for the plot
        """
        # Extract data
        paths = results['reference_paths']
        tracking_results = results['tracking_results']
        target_point = results['target_point']
        start_points = results['start_points']
        optimal_tau = results['optimization_results']['optimal_tau']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{title} (τ = {optimal_tau:.3f})')
        
        # Plot 1: Path tracking in SL-SF space
        for i in range(len(paths)):
            ax1.plot(paths[i][:, 6], paths[i][:, 5], '-o', color='blue', linewidth=2, 
                    label=f'Reference Path {i+1}', markersize=4)
            ax1.plot(1/tracking_results[i]['step_times'], tracking_results[i]['step_lengths'], 
                    '-o', color='red', linewidth=2, label=f'Tracked Path {i+1}', markersize=4)
            ax1.plot(start_points[i][1], start_points[i][0], 'go', markersize=8, label=f'Start Point {i+1}')
            ax1.plot(target_point[1], target_point[0], 'ko', markersize=8, label='Target Point')
        
        ax1.set_xlabel('Step Frequency [dimensionless]')
        ax1.set_ylabel('Step Length [dimensionless]')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('Path Tracking in SL-SF Space')
        
        # Plot 2: Time series
        steps = np.arange(len(paths[0]))
        for i in range(len(paths)):
            ax2.plot(steps, paths[i][:, 5], '-o', color='blue', linewidth=2, 
                    label=f'Reference SL {i+1}', markersize=4)
            ax2.plot(steps, tracking_results[i]['step_lengths'], '-o', color='red', linewidth=2, 
                    label=f'Tracked SL {i+1}', markersize=4)
            ax2.plot(steps, paths[i][:, 6], '--o', color='blue', linewidth=2, 
                    label=f'Reference SF {i+1}', markersize=4, alpha=0.7)
            ax2.plot(steps, 1/tracking_results[i]['step_times'], '--o', color='red', linewidth=2, 
                    label=f'Tracked SF {i+1}', markersize=4, alpha=0.7)
        
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Step Length/Frequency [dimensionless]')
        ax2.legend()
        ax2.grid(True)
        ax2.set_title('Time Series')
        
        plt.tight_layout()
        
        # Save plot
        import os
        os.makedirs('logs/plots', exist_ok=True)
        plt.savefig(f'logs/plots/{title.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
