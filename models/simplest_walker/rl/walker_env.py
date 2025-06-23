"""
Custom Gym environment for the simplest walker model.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from models.simplest_walker.SimplestWalker import SimplestWalker
from models.simplest_walker.analysis.utils import (load_limit_cycle_solutions,
                                                walker_state_interpol,
                                                feedback_gain_interpol)


class WalkerEnv(gym.Env):
    """Custom Environment for the simplest walker model."""

    def __init__(self, task=None):
        """
        Initialize the environment.

        Args:
            task: Optional task path to follow. If None, no task will be set and the environment
                 will wait for a task to be set (typically by a wrapper).
        """
        super().__init__()

        # Define action space (required by DDPG). this should be [-1, 1] because of the tanh activation in the policy
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),  # Standard DDPG range
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # Define observation space: [target_sl, target_sf, current_sl, current_sf]
        self.observation_space = spaces.Box(
            low=np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
            high=np.array([1.1, 1.1, 1.1, 1.1], dtype=np.float32),
            dtype=np.float32
        )

        # Initialize walker
        self.walker = None
        
        # Set task if provided
        self.task = task
        self.current_step = 0
        self.max_steps = len(self.task) - 1 if self.task is not None else 0

    def reset(self, seed=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize walker with first point
        starting_point = self.task[0]
        current_sl   = starting_point[5]
        current_sf   = starting_point[6]
        x0 = starting_point[:2]
        s_nominal = starting_point[0:2]
        u_nominal = starting_point[2:5]
        self.walker = SimplestWalker(x0, s_nominal, u_nominal)
        
        self.current_step = 0
        target_point = self.task[self.current_step+1]
       
        # Initial observation: [target_sl, target_sf, current_sl, current_sf]
        observation = np.array([
            target_point[5],  # target step length
            target_point[6],  # target step frequency
            current_sl,       # current step length
            current_sf        # current step frequency
        ], dtype=np.float32)

        return observation, {}

    def step(self, action):
        """
        Take one step in the environment.

        Args:
            action: Control inputs [pushoff, hip_stiffness, hip_stiffness] in range [-1, 1]

        Returns:
            observation: Next state observation
            reward: Reward for the step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # to remove the tanh effect from the policy (set by default in SB3)
        # Clip actions to prevent numerical instability
        action = np.clip(action, -0.999, 0.999)
        action_arctanh = np.arctanh(action)  # More stable than log-based formula

        target_point = self.task[self.current_step]
        x0 = self.walker.x0[:2]  # Current state

        # Take one step
        next_s, _, _ = self.walker.take_one_step(x0, action_arctanh)
        next_s = next_s[:2]

        if self.walker.fall_flag:
            sl = 0
            sf = 0
            reward = 0
            done = False
            truncated = True
            # print("Walker fell down!")
        else:
            # Calculate step measures
            sl, _, _, st = self.walker.get_step_measures(next_s)
            sl = np.clip(sl, 0.01, np.inf)
            st = np.clip(st, 0.01, np.inf)
            sf = 1/st
            
            # Check if we've reached the end of the task
            if self.current_step >= self.max_steps:
                done = True
                truncated = False
            else:
                done = False
                truncated = False
        
            # Calculate reward
            rmse_sl = np.sqrt((sl - target_point[5])**2)
            rmse_sf = np.sqrt((sf - target_point[6])**2)
            reward = 1 - (1*rmse_sl + 10 * rmse_sf)

            # Update state
            self.walker.x0 = next_s
            self.current_step += 1

        # Create observation
        observation = np.array([target_point[5], target_point[6], sl, sf], dtype=np.float32)

        return observation, reward, done, truncated, {}

    def render(self):
        """Render the environment."""
        pass  # Implement if needed

    def close(self):
        """Clean up resources."""
        pass  # Implement if needed 