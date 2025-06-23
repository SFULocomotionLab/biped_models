"""
Entry point for running reinforcement learning training.
"""
import numpy as np
import torch
from models.simplest_walker.analysis.tracking_analysis import TrackingAnalysis
from models.simplest_walker.analysis.NeuralNetworkController import NeuralNetworkController
from models.simplest_walker.rl.train_ddpg import train_ddpg, plot_training_rewards
from stable_baselines3 import DDPG

def main():
    """Run the reinforcement learning training."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pre-trained neural network
    pretrained_model = NeuralNetworkController(n_hidden_layers=2)
    pretrained_model.load_state_dict(
        torch.load('data/simplest_walker/NN_controller.pth', map_location=device)
    )
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()  # Set to evaluation mode
    
    # Create tracking analysis object
    tracking = TrackingAnalysis()
    total_timesteps = 10000
    sine_params = {
            'ampSL': 0.1,
            'ampSF': 0.1,
            'freqSL': 2.0,
            'freqSF': 1.0,
             'time': 1.0,
            'n_steps': 31 # temporary when using domain randomization or curriculum learning
        }

    # Important: switch between single task and multiple task evaluation
    single_task = True
    if single_task:
        curriculum_config=None
        domain_config=None

        # Generate a single sinusoidal path
        start_point = (0.6, 0.6)  # (SL, SF)
        path, _ = tracking.generate_sinusoid_path(start_point, sine_params)
    else:
    # Define curriculum and domain randomization learning configuration
        curriculum_config = {
        'initial_start_point_range': ((0.5, 0.5), (0.7, 0.7)),  # Easier initial range (SL, SF), (SL, SF)
        'final_start_point_range': ((0.35, 0.35), (0.85, 0.85)),  # Harder final range (SL, SF), (SL, SF)
        'initial_n_steps_range': (80, 200),  # More steps initially
        'final_n_steps_range': (30, 100),  # Fewer steps at end
        'curriculum_type': 'episode',  # or 'performance'
        'curriculum_threshold': 0.8,  # for performance-based curriculum
        'curriculum_schedule': 'linear',  # or 'exponential'
        'curriculum_steps': total_timesteps,
        'sine_params': sine_params
        }

        domain_config = {
        'start_point_range': ((0.35, 0.35), (0.85, 0.85)), # (SL, SF), (Sl, SF)
        'n_steps_range': (30, 100),
        'sine_params': sine_params
        }
        path = None

    # Train DDPG agent
    n_parallel = 64
    model = train_ddpg(
        pretrained_model, 
        n_parallel=n_parallel, 
        total_timesteps=total_timesteps,
        curriculum_config=curriculum_config,
        domain_config=None,
        task=path
    )

    #--------------------------------------------------
    # Plot training rewards
    plot_training_rewards("./logs/results/", curriculum_config=None, domain_config=domain_config, n_parallel=n_parallel)
    
    # Load the best model for evaluation
    best_model = DDPG.load("./logs/best_model/best_model")
    best_model.actor = best_model.actor.to(device)  # Move actor to device
    
    if curriculum_config is not None or domain_config is not None:
        # Evaluate on multiple randomized paths
        n_eval_paths = 5
        eval_rewards = []
        
        for i in range(n_eval_paths):
        # Generate a new randomized path for evaluation
        # Use the final difficulty level for evaluation
            eval_start_point = (
                np.random.uniform(0.35, 0.85),  # Random SL from final range
                np.random.uniform(0.35, 0.85)   # Random SF from final range
            )
            eval_n_steps = np.random.randint(30, 101)  # Random number of steps from final range
        
            eval_sine_params = {
                'ampSL': 0.1,
                'ampSF': 0.1,
                'freqSL': 2.0,
                'freqSF': 1.0,
                'time': 1.0,
                'n_steps': eval_n_steps
                 }
        
            eval_path, _ = tracking.generate_sinusoid_path(eval_start_point, eval_sine_params)
        
            # Evaluate on this path
            eval_results = tracking.track_path(eval_path, [], neural_net=best_model.actor, RLtrained=True)
            eval_rewards.append(np.sum(eval_results['rewards']))
        
            # Plot results for this evaluation path
            tracking.plot_results(
                eval_path, 
                eval_results, 
                title=f'After RL training (Eval {i+1})'
            )
    
         # Plot average evaluation performance
        print(f'Average reward across {n_eval_paths} evaluation paths: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}')
    else: # single task evaluation
        eval_start_point = start_point
        eval_sine_params = sine_params
        eval_path, _ = tracking.generate_sinusoid_path(eval_start_point, eval_sine_params)
        eval_results = tracking.track_path(eval_path, [], neural_net=best_model.actor, RLtrained=True)
        tracking.plot_results(eval_path, eval_results, title='After RL training')
        print('reward after training: ', np.sum(eval_results['rewards']))
        
        # Plot neural network performance before RL training on original path
        nn_results = tracking.track_path(path, [], neural_net=pretrained_model, RLtrained=False)
        tracking.plot_results(path, nn_results, title='Before RL training')
        print('reward before training: ', np.sum(nn_results['rewards']))


if __name__ == "__main__":
    main() 