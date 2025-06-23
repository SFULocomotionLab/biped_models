"""
Maximum Tolerable Perturbation Analysis

This module calculates the maximum tolerable perturbation 
for each gait in the gait space. It starts by trying to converge from neighboring 
points to the target point, and if all neighboring gaits converge without falling 
and can walk for 100 steps, it goes to the next layer of gaits. It continues this 
until it can't converge, and the radius of the layer where all gaits converged 
to the target is the maximum tolerable perturbation.

The analysis is performed for two scenarios: with and without feedback control.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import time
import warnings

from models.simplest_walker.SimplestWalker import SimplestWalker
from models.simplest_walker.analysis.utils import (
    load_limit_cycle_solutions, 
    load_linear_analysis_data
)


class MaximumTolerablePerturbation:
    """
    Class to calculate maximum tolerable perturbation
    for gaits in the gait space.
    """
    
    def __init__(self, 
                 solutions_file: str = 'data/simplest_walker/limit_cycle_solutions.npz',
                 linear_analysis_file: str = 'data/simplest_walker/linear_analysis_results.npz'):
        """
        Initialize the maximum tolerable perturbation analysis.
        
        Args:
            solutions_file: Path to limit cycle solutions file
            linear_analysis_file: Path to linear analysis results file
        """
        # Load data
        self.solutions, self.target_step_lengths, self.target_step_frequencies = \
            load_limit_cycle_solutions(solutions_file)
        
        self.linear_analysis_data = load_linear_analysis_data(linear_analysis_file)
        
        # Analysis parameters
        self.n_steps = 100  # Number of steps to test convergence
        self.min_radius = 0.01  # Minimum radius to start with
        self.step_size = 0.01  # Step size for increasing radius
        self.max_radius = 0.5  # Maximum radius to prevent infinite loops
        
        # Results storage
        self.max_tolerable_perturbation_with_control = None
        self.max_tolerable_perturbation_without_control = None
        
    def _get_gait_state_and_control(self, sl_idx: int, sf_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the stored state and control for a specific gait from solutions.
        
        Args:
            sl_idx: Step length index
            sf_idx: Step frequency index
            
        Returns:
            Tuple of (state, control) where state is [stance_angle, stance_vel] and 
            control is [pushoff, k1, k2]
        """
        # Get the solution for this gait
        solution = self.solutions[0][sf_idx][sl_idx]
        
        # Extract state (first 2 columns: stance_angle, stance_vel)
        state = solution[:2]
        
        # Extract control (columns 2-4: pushoff, k1, k2)
        control = solution[2:4]
        control = np.append(control, control[1])
        
        return state, control
        
    def calculate_max_tolerable_perturbation_for_gait(self, 
                                                    sl_idx: int, 
                                                    sf_idx: int, 
                                                    with_control: bool = True) -> float:
        """
        Calculate maximum tolerable perturbation for a specific gait.
        
        Args:
            sl_idx: Step length index
            sf_idx: Step frequency index  
            with_control: Whether to use feedback control
            
        Returns:
            Maximum tolerable perturbation radius
        """
        # Get target gait parameters
        target_sl = self.target_step_lengths[sl_idx]
        target_sf = self.target_step_frequencies[sf_idx]
        
        # Get target state and control from stored solutions
        target_state, target_control = self._get_gait_state_and_control(sl_idx, sf_idx)
        
        # Get feedback gains if using control
        if with_control:
            # Use the stored feedback gains for this specific gait
            target_K = self.linear_analysis_data['K_pp'][sl_idx, sf_idx]
        else:
            target_K = np.zeros((3, 2))
        
        # Create walker object
        walker = SimplestWalker(
            x0=target_state,  # Use stored state directly
            s_nominal=target_state,
            u_nominal=target_control
        )
        
        # Initialize radius
        radius = self.min_radius
        max_radius_reached = False
        
        while not max_radius_reached and radius <= self.max_radius:
            # Find all points within current radius
            points_within_radius = self._find_points_within_radius(
                target_sl, target_sf, radius
            )
            
            if len(points_within_radius) == 0:
                # No points found, increase radius
                radius += self.step_size
                continue
            
            # Debug: Print information about current radius test
            if radius < 0.05:  # Only print for small radii to avoid spam
                print(f"  Testing radius {radius:.3f} with {len(points_within_radius)} points")
            
            # Test convergence for all points within radius
            all_converged = True
            failed_points = []
            for point_sl, point_sf in points_within_radius:
                if not self._test_convergence(
                    point_sl, point_sf, target_sl, target_sf, 
                    target_control, target_K, walker, with_control
                ):
                    all_converged = False
                    failed_points.append((point_sl, point_sf))
                    break
            
            if not all_converged:
                # Some points didn't converge, return previous radius
                if radius < 0.05:  # Debug info
                    print(f"    Failed at radius {radius:.3f}, failed points: {failed_points[:2]}...")
                return radius - self.step_size
            else:
                # All points converged, increase radius
                radius += self.step_size
        
        return radius - self.step_size
    
    def _find_points_within_radius(self, 
                                 target_sl: float, 
                                 target_sf: float, 
                                 radius: float) -> List[Tuple[float, float]]:
        """
        Find all gait points within a given radius of the target point.
        
        Args:
            target_sl: Target step length
            target_sf: Target step frequency
            radius: Radius to search within
            
        Returns:
            List of (step_length, step_frequency) tuples within radius
        """
        points_within_radius = []
        
        for i, sl in enumerate(self.target_step_lengths):
            for j, sf in enumerate(self.target_step_frequencies):
                distance = np.sqrt((sl - target_sl)**2 + (sf - target_sf)**2)
                # Exclude the target point itself and only include points within radius
                if distance <= radius + 1e-8 and distance > 1e-8:  # Small tolerance for floating point
                    points_within_radius.append((sl, sf))
        
        return points_within_radius
    
    def _test_convergence(self, 
                         start_sl: float, 
                         start_sf: float, 
                         target_sl: float, 
                         target_sf: float,
                         target_control: np.ndarray, 
                         target_K: np.ndarray, 
                         walker: SimplestWalker,
                         with_control: bool) -> bool:
        """
        Test if a gait converges to the target gait within n_steps.
        Matches MATLAB implementation: only check if walker falls, not convergence to target.
        
        Args:
            start_sl: Starting step length
            start_sf: Starting step frequency
            target_sl: Target step length
            target_sf: Target step frequency
            target_control: Target control parameters
            target_K: Target feedback gains
            walker: Walker object
            with_control: Whether to use feedback control
            
        Returns:
            True if gait doesn't fall, False otherwise
        """
        # Find the indices for the starting gait
        sl_idx = np.argmin(np.abs(self.target_step_lengths - start_sl))
        sf_idx = np.argmin(np.abs(self.target_step_frequencies - start_sf))
            
        # Get starting state from stored solutions
        start_state, _ = self._get_gait_state_and_control(sl_idx, sf_idx)
            
        # Initialize walker with starting state
        current_state = start_state.copy()
            
        # Simulate for n_steps
        for step in range(self.n_steps):
            # Apply feedback control if enabled
            if with_control:
                # Use the walker's feedback controller method
                current_control = walker.apply_feedback_controller(current_state, target_K)
            else:
                current_control = target_control
                
            # Take one step
            next_state, _, _ = walker.take_one_step(current_state, current_control)
                
            # Check if walker fell (MATLAB logic: only check falling)
            if walker.fall_flag:
                return False
                
            # Update state for next step
            current_state = next_state[:2]
            
        # If we reach here, the walker didn't fall (MATLAB success criteria)
        return True
            
    
    def calculate_max_tolerable_perturbation_grid(self, 
                                                with_control: bool = True) -> np.ndarray:
        """
        Calculate maximum tolerable perturbation for all gaits in the grid.
        
        Args:
            with_control: Whether to use feedback control
            
        Returns:
            Matrix of maximum tolerable perturbation values
        """
        n_sl = len(self.target_step_lengths)
        n_sf = len(self.target_step_frequencies)
        
        max_tolerable_perturbation = np.full((n_sl, n_sf), np.nan)
        
        # Sequential processing
        for sl_idx in range(n_sl):
            for sf_idx in range(n_sf):
                try:
                    result = self.calculate_max_tolerable_perturbation_for_gait(
                        sl_idx, sf_idx, with_control
                    )
                    max_tolerable_perturbation[sl_idx, sf_idx] = result
                    print(f"Completed gait ({sl_idx}, {sf_idx}): {result:.3f}")
                except Exception as e:
                    print(f"Error processing gait ({sl_idx}, {sf_idx}): {e}")
        
        return max_tolerable_perturbation
    
    def run_analysis(self):
        """
        Run the complete maximum tolerable perturbation analysis.
        """
        print("Starting maximum tolerable perturbation analysis...")
        
        # Analysis with feedback control
        print("\nCalculating maximum tolerable perturbation WITH feedback control...")
        start_time = time.time()
        self.max_tolerable_perturbation_with_control = \
            self.calculate_max_tolerable_perturbation_grid(
                with_control=True
            )
        print(f"Completed with control in {time.time() - start_time:.2f} seconds")
        
        # Analysis without feedback control
        print("\nCalculating maximum tolerable perturbation WITHOUT feedback control...")
        start_time = time.time()
        self.max_tolerable_perturbation_without_control = \
            self.calculate_max_tolerable_perturbation_grid(
                with_control=False
            )
        print(f"Completed without control in {time.time() - start_time:.2f} seconds")
        
        print("\nAnalysis completed!")
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot the maximum tolerable perturbation results.
        
        Args:
            save_path: Optional path to save the figure
        """
        if (self.max_tolerable_perturbation_with_control is None or 
            self.max_tolerable_perturbation_without_control is None):
            raise ValueError("Analysis must be run before plotting results")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot with feedback control
        self._plot_perturbation_circles(
            ax1, self.max_tolerable_perturbation_with_control,
            "Maximum Tolerable Perturbation\nWITH Feedback Control"
        )
        
        # Plot without feedback control
        self._plot_perturbation_circles(
            ax2, self.max_tolerable_perturbation_without_control,
            "Maximum Tolerable Perturbation\nWITHOUT Feedback Control"
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_perturbation_circles(self, ax, perturbation_matrix: np.ndarray, title: str):
        """
        Plot perturbation circles for a given matrix.
        
        Args:
            ax: Matplotlib axis
            perturbation_matrix: Matrix of perturbation values
            title: Plot title
        """
        # Create meshgrid for plotting - swap order to get correct orientation
        SF, SL = np.meshgrid(self.target_step_frequencies, self.target_step_lengths)
        
        # Debug: Print some statistics about the perturbation matrix
        valid_radii = perturbation_matrix[~np.isnan(perturbation_matrix)]
        if len(valid_radii) > 0:
            print(f"Perturbation statistics for {title}:")
            print(f"  Min radius: {np.min(valid_radii):.4f}")
            print(f"  Max radius: {np.max(valid_radii):.4f}")
            print(f"  Mean radius: {np.mean(valid_radii):.4f}")
            print(f"  Number of valid points: {len(valid_radii)}")
        
        # Plot circles for each gait
        for i in range(len(self.target_step_lengths)):
            for j in range(len(self.target_step_frequencies)):
                radius = perturbation_matrix[i, j]
                if not np.isnan(radius) and radius > 0:
                    # Create circle
                    theta = np.linspace(0, 2*np.pi, 100)
                    x_circle = SF[i, j] + radius * np.cos(theta)
                    y_circle = SL[i, j] + radius * np.sin(theta)
                    
                    # Plot circle with better visibility - use different colors based on radius
                    # Normalize radius for color mapping
                    if len(valid_radii) > 0:
                        normalized_radius = (radius - np.min(valid_radii)) / (np.max(valid_radii) - np.min(valid_radii))
                        color = plt.cm.viridis(normalized_radius)
                    else:
                        color = 'blue'
                    
                    # Plot circle with more visible styling
                    ax.fill(x_circle, y_circle, color=color, alpha=0.3, edgecolor='black', linewidth=0.5)
                    
                    # Add radius text for debugging (only for a few points to avoid clutter)
                    if i % 3 == 0 and j % 3 == 0:
                        ax.text(SF[i, j], SL[i, j], f'{radius:.3f}', 
                               fontsize=6, ha='center', va='center')
        
        # Plot gait points
        # ax.scatter(SF, SL, c='red', s=20, alpha=0.8, zorder=5)
        
        # Set labels and title
        ax.set_xlabel('Step frequency [non-dimensional]')
        ax.set_ylabel('Step length [non-dimensional]')
        ax.set_title(title)
        ax.set_xlim([0.1, 1.1])
        ax.set_ylim([0.1, 1.1])
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add colorbar if there are valid radii
        if len(valid_radii) > 0:
            norm = plt.Normalize(np.min(valid_radii), np.max(valid_radii))
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Perturbation Radius')
        

    
    def save_results(self, file_path: str):
        """
        Save the analysis results to a file.
        
        Args:
            file_path: Path to save the results
        """
        np.savez(
            file_path,
            max_tolerable_perturbation_with_control=self.max_tolerable_perturbation_with_control,
            max_tolerable_perturbation_without_control=self.max_tolerable_perturbation_without_control,
            target_step_lengths=self.target_step_lengths,
            target_step_frequencies=self.target_step_frequencies
        )
        print(f"Results saved to {file_path}")
    
    def load_results(self, file_path: str):
        """
        Load analysis results from a file.
        
        Args:
            file_path: Path to load the results from
        """
        data = np.load(file_path)
        self.max_tolerable_perturbation_with_control = data['max_tolerable_perturbation_with_control']
        self.max_tolerable_perturbation_without_control = data['max_tolerable_perturbation_without_control']
        print(f"Results loaded from {file_path}")
