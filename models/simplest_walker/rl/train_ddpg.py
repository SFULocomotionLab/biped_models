"""
Train a DDPG agent for the simplest walker model using a pre-trained neural network.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import torch.multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.monitor import load_results

from models.simplest_walker.rl.walker_env import WalkerEnv
from models.simplest_walker.rl.SinePathRandomizationWrapper import SinePathRandomizationWrapper
from models.simplest_walker.analysis.tracking_analysis import TrackingAnalysis


class CustomCritic(nn.Module):
    """Custom critic network for DDPG."""
    
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.obs_net = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.Sigmoid(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.Sigmoid(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        
        self.combined_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1)
        )
    
    def forward(self, obs, action):
        obs_features = self.obs_net(obs)
        action_features = self.action_net(action)
        combined = torch.cat([obs_features, action_features], dim=1)
        return self.combined_net(combined)


class PretrainedActorPolicy(MlpPolicy):
    def __init__(self, *args, pretrained_actor=None, **kwargs):
        # Override the default activation function to use Sigmoid
        if 'net_arch' not in kwargs:
            kwargs['net_arch'] = dict(
                pi=[64, 64],
                qf=[64, 64]
            )
        if 'activation_fn' not in kwargs:
            kwargs['activation_fn'] = nn.Sigmoid
            
        super().__init__(*args, **kwargs)

        if pretrained_actor is not None:
            # Move pretrained actor to the same device as this model
            device = next(self.parameters()).device
            pretrained_actor = pretrained_actor.to(device)
            
            # Rename keys from network.* to mu.*
            renamed_state_dict = _rename_prefix(pretrained_actor.state_dict())
            
            # Load the state dict
            self.actor.load_state_dict(renamed_state_dict)
            
            # Remove the tanh activation from the actor's output
            # self.actor.mu = nn.Sequential(*list(self.actor.mu.children())[:-1])

    def forward(self, obs, deterministic=False):
        """
        Forward pass.
        
        Args:
            obs: Observation tensor
            deterministic: Whether to use deterministic actions
            
        Returns:
            Raw actions from the actor network
        """
        # Get actions from actor
        actions = self.actor(obs)
        
        # Add noise if not deterministic
        if not deterministic and self.action_noise is not None:
            actions = actions + torch.tensor(self.action_noise(), device=actions.device)
        
        return actions


def _rename_prefix(state_dict):
    """
    Renames all keys from network.* to mu.* format.
    Example: 'network.0.weight' → 'mu.0.weight'
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        # Split the key into parts
        parts = k.split('.')
        if parts[0] == 'network':
            # Replace 'network' with 'mu' and keep the rest of the key
            new_key = 'mu.' + '.'.join(parts[1:])
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def make_env(rank, seed=0, curriculum_config=None, domain_config=None, task=None):
    """
    Create a wrapped environment.
    
    Args:
        rank: Process rank
        seed: Random seed
        curriculum_config: Optional dictionary containing curriculum learning parameters
        domain_config: Optional dictionary containing domain randomization parameters
        task: Optional task path to follow. Only used if no randomization configs are provided.
    """
    def _init():
        try:
            # Create base environment
            # Only pass task if no randomization is being used
            env = WalkerEnv(task=task if not (curriculum_config or domain_config) else None)
            
            # wrap with curriculum or domain randomization if provided
            if curriculum_config or domain_config:
                # Wrap with either curriculum learning or domain randomization
                env = SinePathRandomizationWrapper(env, curriculum_config=curriculum_config, domain_config=domain_config)
            
            # Create a unique filename for each environment's monitor
            monitor_path = os.path.join("logs", "results", f"monitor_{rank}")
            env = Monitor(env, monitor_path)
            env.reset(seed=seed + rank)
            return env
        except Exception as e:
            print(f"Error creating environment {rank}: {str(e)}")
            raise
    return _init

def train_ddpg(pretrained_actor, n_parallel=None, total_timesteps=250, curriculum_config=None, domain_config=None, hyperparams=None, task=None):
    """
    Train a DDPG agent using a pre-trained actor network.
    
    Args:
        pretrained_actor: Pre-trained neural network for the actor
        n_parallel: Number of parallel environments (default: None, will use CPU count - 1)
        total_timesteps: Total number of timesteps to train for
        curriculum_config: Optional dictionary containing curriculum learning parameters
        domain_config: Optional dictionary containing domain randomization parameters
        hyperparams: Optional dictionary containing hyperparameters to override defaults
        task: Optional task path to follow. If provided, will use this specific task for training.
             If None and no randomization configs provided, will use default task.
    """
    print("Starting DDPG training...")
    if curriculum_config:
        print("Using curriculum learning with configuration:")
        print(f'Initial start point range: {curriculum_config["initial_start_point_range"]}')
        print(f'Final start point range: {curriculum_config["final_start_point_range"]}')
        print(f'Initial n steps range: {curriculum_config["initial_n_steps_range"]}')
        print(f'Final n steps range: {curriculum_config["final_n_steps_range"]}')
        print(f'Curriculum type: {curriculum_config["curriculum_type"]}')
        print(f'Curriculum schedule: {curriculum_config["curriculum_schedule"]}')
        
        # Calculate expected number of episodes based on curriculum
        avg_episode_length = (curriculum_config['initial_n_steps_range'][0] + curriculum_config['initial_n_steps_range'][1]) / 2
        expected_episodes = total_timesteps / avg_episode_length
        curriculum_config['curriculum_episodes'] = int(expected_episodes)
        print(f"\nTraining Configuration:")
        print(f"Total timesteps for training: {total_timesteps}")
        print(f"Average steps per episode: {avg_episode_length:.1f}")
        print(f"Expected number of episodes: {expected_episodes:.1f}")
        print(f"Curriculum episodes: {curriculum_config['curriculum_episodes']}")
    elif domain_config:
        print("Using domain randomization with configuration:")
        print(f'Start point range: {domain_config["start_point_range"]}')
        print(f'Number of steps range: {domain_config["n_steps_range"]}')
        print(f'Sine parameters:')
        for param, value in domain_config['sine_params'].items():
            print(f'  {param}: {value}')
    elif task is not None:
        print("Using single task")

    # Set number of parallel environments based on CPU cores if not specified
    if n_parallel is None:
        n_parallel = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {n_parallel} parallel environments")
    
    # Create log directories with absolute paths
    log_dir = os.path.abspath("./logs")
    best_model_dir = os.path.join(log_dir, "best_model")
    results_dir = os.path.join(log_dir, "results")
    
    # Ensure directories exist
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Set PyTorch to use multiple threads
    torch.set_num_threads(n_parallel)
    
    # Create vectorized environments with proper error handling
    try:
        if n_parallel > 1:
            env = SubprocVecEnv([make_env(rank=i, curriculum_config=curriculum_config, domain_config=domain_config, task=task) for i in range(n_parallel)])
        else:
            env = DummyVecEnv([make_env(rank=0, curriculum_config=curriculum_config, domain_config=domain_config, task=task)])
        print("Environment created successfully")
    except Exception as e:
        print(f"Error creating environments: {str(e)}")
        raise

    # Create evaluation environment
    eval_env = WalkerEnv(task=task if not (curriculum_config or domain_config) else None)
    if curriculum_config or domain_config:
        eval_env = SinePathRandomizationWrapper(eval_env, curriculum_config=curriculum_config, domain_config=domain_config)
    eval_env = Monitor(eval_env, results_dir)
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=results_dir,
        eval_freq=50,
        deterministic=True,
        render=False,
        verbose=0
    )

    # Initialize action noise
    n_actions = env.action_space.shape[0]
    noise_sigma = hyperparams.get('noise_sigma', 0.001) if hyperparams else 0.004
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=noise_sigma * np.ones(n_actions),
    )
    
    # Get hyperparameters with defaults
    buffer_size = hyperparams.get('buffer_size', 512) if hyperparams else 512
    batch_size = hyperparams.get('batch_size', 256) if hyperparams else 128
    learning_rate = hyperparams.get('learning_rate', 1e-7) if hyperparams else 1e-6
    tau = hyperparams.get('tau', 0.001) if hyperparams else 0.001
    
    print("Creating DDPG model...")

    # Create and train the agent
    model = DDPG(
        policy=PretrainedActorPolicy,
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=1,
        batch_size=batch_size,
        tau=tau,
        gamma=0.99,
        train_freq=1,  # Train every n steps
        gradient_steps=-1,
        action_noise=action_noise,
        policy_kwargs=dict(pretrained_actor=pretrained_actor),
        verbose=2
    )
    
    print("Starting model training...")
    
    try:
        # Train the agent with both callbacks
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback]
        )
        print("Training completed successfully")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    finally:
        # Clean up parallel environments
        env.close()
        eval_env.close()
    
    return model

def plot_training_rewards(model_log_path, curriculum_config=None, domain_config=None, n_parallel=None):
    """
    Plot training rewards over total steps with vertical lines indicating curriculum stages.
    """
    try:
        # Load results from all parallel environments
        all_episodes = []
        
        # Collect all episodes from all environments with their timestamps
        for i in range(n_parallel if n_parallel>1 else 1):
            monitor_path = os.path.join(model_log_path, f"monitor_{i}.monitor.csv")
            if os.path.exists(monitor_path):
                env_data = pd.read_csv(monitor_path, skiprows=1)
                if not env_data.empty:
                    # Add environment ID
                    env_data['env_id'] = i
                    all_episodes.append(env_data)
        
        if not all_episodes:
            print("No training data found!")
            return
            
        # Combine all episodes into a single DataFrame
        combined_data = pd.concat(all_episodes)
        
        # Sort episodes by their actual timestamp
        combined_data = combined_data.sort_values('t')
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot rewards over steps
        cumulative_steps = np.cumsum(combined_data['l'])
        plt.plot(cumulative_steps, combined_data['r'], alpha=0.3)
        
        # Add vertical lines for curriculum stages if curriculum_config is provided
        if curriculum_config:
            total_episodes = curriculum_config['curriculum_episodes']
            steps_per_stage = cumulative_steps.iloc[-1] / total_episodes
            
            for stage in range(1, total_episodes + 1):
                stage_step = stage * steps_per_stage
                plt.axvline(x=stage_step, color='r', linestyle='--', alpha=0.3)
                if stage % 5 == 0:  # Label every 5th stage 
                    plt.text(stage_step, plt.ylim()[1], f'Stage {stage}', 
                            rotation=90, verticalalignment='bottom')
        
        plt.xlabel('Total Steps')
        plt.ylabel('Reward')
        plt.title('Training Rewards Over Total Steps')
        plt.grid(True)
        ymin = max(-1, combined_data['r'].min())
        ymax = combined_data['r'].max() + 1
        plt.ylim(ymin, ymax)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('logs/plots', exist_ok=True)
        plt.savefig(f'logs/plots/training_rewards.png')
        plt.close()
        
        print(f'total steps: {cumulative_steps.iloc[-1]}')
        print(f'total episodes: {len(combined_data)}')
        print(f'steps per episode: {cumulative_steps.iloc[-1] / len(combined_data)}')
        
    except Exception as e:
        print(f"Error plotting training rewards: {str(e)}")
        raise

    