"""
Domain randomization wrapper for the simplest walker environment with curriculum learning.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, Tuple


class SinePathRandomizationWrapper(gym.Wrapper):
    """
    A wrapper that adds either curriculum learning or domain randomization to the walker environment.
    This wrapper randomizes the sine path task parameters and optionally gradually increases difficulty.
    """
    
    def __init__(
        self,
        env: gym.Env,
        curriculum_config: Optional[Dict[str, Any]] = None,
        domain_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the wrapper with either curriculum learning or domain randomization.

        Args:
            env: The environment to wrap
            curriculum_config: Dictionary containing curriculum learning parameters
                - initial_start_point_range: Initial range for starting points
                - final_start_point_range: Final range for starting points
                - initial_n_steps_range: Initial range for number of steps
                - final_n_steps_range: Final range for number of steps
                - curriculum_type: 'episode' or 'performance'
                - curriculum_threshold: Performance threshold for 'performance' type
                - curriculum_steps: Total number of steps for curriculum progression
                - curriculum_schedule: 'linear' or 'exponential' for difficulty progression
            domain_config: Dictionary containing domain randomization parameters
                - start_point_range: Tuple of (min, max) for starting point randomization [SL, SF]
                - n_steps_range: Tuple of (min, max) for number of steps randomization
                - sine_params: Dictionary of fixed sine wave parameters
                    - ampSL: Amplitude for stride length
                    - ampSF: Amplitude for stride frequency
                    - freqSL: Frequency for stride length
                    - freqSF: Frequency for stride frequency
                    - time: Time parameter
        """
        super().__init__(env)
        
        # Default curriculum configuration
        self.default_curriculum = {
            'initial_start_point_range': ((0.5, 0.5), (0.7, 0.7)),  # Easier initial range
            'final_start_point_range': ((0.35, 0.35), (0.85, 0.85)),  # Harder final range
            'initial_n_steps_range': (50, 100),  # More steps initially
            'final_n_steps_range': (30, 100),  # Fewer steps at end
            'curriculum_type': 'episode',  # 'episode' or 'performance'
            'curriculum_threshold': 0.8,  # Performance threshold for 'performance' type
            'curriculum_steps': 10000,  # Total number of steps for curriculum
            'curriculum_schedule': 'linear'  # 'linear' or 'exponential'
        }
        
        # Default domain randomization configuration
        self.default_domain = {
            'start_point_range': ((0.35, 0.35), (0.85, 0.85)),  # Range for [SL, SF] starting point
            'n_steps_range': (20, 100),  # Range for number of steps
            'sine_params': {
                'ampSL': 0.1,  # Fixed amplitude for stride length
                'ampSF': 0.1,  # Fixed amplitude for stride frequency
                'freqSL': 2.0,  # Fixed frequency for stride length
                'freqSF': 1.0,  # Fixed frequency for stride frequency
                'time': 1.0    # Fixed time
            }
        }
        
        # Determine which mode to use
        if curriculum_config and domain_config:
            raise ValueError("Cannot use both curriculum learning and domain randomization simultaneously")
        elif curriculum_config:
            self.mode = 'curriculum'
            self.config = self.default_curriculum.copy()
            self.config.update(curriculum_config)
        elif domain_config:
            self.mode = 'domain'
            self.config = self.default_domain.copy()
            self.config.update(domain_config)
        else:
            raise ValueError("Must provide either curriculum_config or domain_config")
            
        self.total_steps = 0
        
        # Curriculum learning state (only used in curriculum mode)
        if self.mode == 'curriculum':
            self.current_difficulty = 0.0  # 0.0 to 1.0
            
        # Store starting points for each episode
        self.starting_points = []
        
        # Initialize tracking analysis
        from models.simplest_walker.analysis.tracking_analysis import TrackingAnalysis
        self.tracking = TrackingAnalysis()
        
        # Generate initial task path
        self._randomize_parameters()
        
    def get_starting_points(self):
        """Return the list of starting points used during training."""
        return np.array(self.starting_points)
        
    def _update_difficulty(self):
        """Update the current difficulty level based on curriculum type."""
        if self.mode != 'curriculum':
            return
            
        if self.config['curriculum_type'] == 'episode':
            # Simple linear progression based on total steps
            constant_term = 10
            self.current_difficulty = min(1.0, constant_term * self.total_steps / self.config['curriculum_steps'])
            
        elif self.config['curriculum_type'] == 'performance':
            # Update based on performance history
            if len(self.performance_history) > 0:
                recent_performance = np.mean(self.performance_history[-10:])  # Last 10 episodes
                if recent_performance >= self.config['curriculum_threshold']:
                    self.current_difficulty = min(1.0, self.current_difficulty + 0.1)
        
        # Apply schedule type
        if self.config['curriculum_schedule'] == 'exponential':
            self.current_difficulty = np.exp(self.current_difficulty - 1)
            
    def _get_current_ranges(self):
        """Get the current parameter ranges based on mode and difficulty level."""
        if self.mode == 'curriculum':
            # Simple linear interpolation between initial and final ranges
            start_point_min = np.array(self.config['initial_start_point_range'][0]) + \
                            self.current_difficulty * (np.array(self.config['final_start_point_range'][0]) - 
                                                     np.array(self.config['initial_start_point_range'][0]))
            start_point_max = np.array(self.config['initial_start_point_range'][1]) + \
                            self.current_difficulty * (np.array(self.config['final_start_point_range'][1]) - 
                                                     np.array(self.config['initial_start_point_range'][1]))
            # print(f" difficulty {self.current_difficulty:.2f}:")
            # print(f"  start_point_min, start_point_max: {start_point_min}, {start_point_max}")

            n_steps_min = int(self.config['initial_n_steps_range'][0] + \
                            self.current_difficulty * (self.config['final_n_steps_range'][0] - 
                                                     self.config['initial_n_steps_range'][0]))
            n_steps_max = int(self.config['initial_n_steps_range'][1] + \
                            self.current_difficulty * (self.config['final_n_steps_range'][1] - 
                                                     self.config['initial_n_steps_range'][1]))
        else:  # domain mode
            start_point_min = np.array(self.config['start_point_range'][0])
            start_point_max = np.array(self.config['start_point_range'][1])
            n_steps_min = self.config['n_steps_range'][0]
            n_steps_max = self.config['n_steps_range'][1]
        
        return (start_point_min, start_point_max), (n_steps_min, n_steps_max)
        
    def _randomize_parameters(self):
        """Randomize the sine path task parameters within specified ranges."""
        
        # Get current ranges
        (start_point_min, start_point_max), (n_steps_min, n_steps_max) = self._get_current_ranges()
        
        # Randomize starting point
        new_start_sl = np.random.uniform(start_point_min[0], start_point_max[0])
        new_start_sf = np.random.uniform(start_point_min[1], start_point_max[1])
        print(f'new_start_sl, new_start_sf: {new_start_sl}, {new_start_sf}')
        
        # Randomize number of steps
        new_n_steps = np.random.randint(n_steps_min, n_steps_max + 1)
        
        sine_params = self.config['sine_params'].copy()
        sine_params['n_steps'] = new_n_steps
        
        # Generate new task path
        new_path, _ = self.tracking.generate_sinusoid_path(
            start_point=(new_start_sl, new_start_sf),
            sine_params=sine_params
        )
        
        # Update environment task
        self.env.task = new_path
        self.env.max_steps = len(new_path) - 1
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment and randomize parameters.
        
        Returns:
            observation: The initial observation
            info: Additional information
        """
        self._randomize_parameters()
        
        # Store the starting point
        start_point = self.env.task[0, 5:7]  # [SL, SF] from first point
        self.starting_points.append(start_point)
        
        return self.env.reset(**kwargs)
        
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            observation: The next observation
            reward: The reward for the step
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update total steps in curriculum mode
        if self.mode == 'curriculum':
            self.total_steps += 1
            self._update_difficulty()  # Update difficulty after each step
        
        # Add starting points to info
        info['start_point'] = self.env.task[0, 5:7]  # [SL, SF] from first point
        
        # Update curriculum state if episode is done
        if (terminated or truncated) and self.mode == 'curriculum':
            if self.config['curriculum_type'] == 'performance':
                self.performance_history.append(reward)
        
        return obs, reward, terminated, truncated, info 