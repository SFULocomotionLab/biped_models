"""
Hyperparameter optimization for DDPG training using Optuna.
"""
import os
import optuna
import numpy as np
import torch
from models.simplest_walker.analysis.tracking_analysis import TrackingAnalysis
from models.simplest_walker.analysis.NeuralNetworkController import NeuralNetworkController
from models.simplest_walker.rl.train_ddpg import train_ddpg
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

def objective(trial, training_mode):
    """Objective function for Optuna optimization."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained neural network
    pretrained_model = NeuralNetworkController(n_hidden_layers=2)
    pretrained_model.load_state_dict(
        torch.load('data/simplest_walker/NN_controller.pth', map_location=device)
    )
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()
    
    # Create tracking analysis object
    tracking = TrackingAnalysis()
    
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 512)
    noise_sigma = trial.suggest_float('noise_sigma', 0.0001, 0.01, log=True)
    buffer_size = trial.suggest_int('buffer_size', 512, 2048)
    tau = trial.suggest_float('tau', 0.001, 0.01, log=True)
    
    # Define sine parameters
    sine_params = {
        'ampSL': 0.1,
        'ampSF': 0.1,
        'freqSL': 2.0,
        'freqSF': 1.0,
        'time': 1.0,
    }
    
    # Set up training configuration based on mode
    if training_mode == 'single_task':
        curriculum_config = None
        domain_config = None
        # Generate a single sinusoidal path for single task training
        start_point = (0.6, 0.6)  # (SL, SF)
        path, _ = tracking.generate_sinusoid_path(start_point, {**sine_params, 'n_steps': 31})
    else:
        path = None
        if training_mode == 'curriculum':
            curriculum_config = {
                'initial_start_point_range': ((0.5, 0.5), (0.7, 0.7)),  # Easier initial range
                'final_start_point_range': ((0.35, 0.35), (0.85, 0.85)),  # Harder final range
                'initial_n_steps_range': (80, 200),  # More steps initially
                'final_n_steps_range': (30, 100),  # Fewer steps at end
                'curriculum_type': 'episode',  # or 'performance'
                'curriculum_threshold': 0.8,  # for performance-based curriculum
                'curriculum_schedule': 'linear',  # or 'exponential'
                'curriculum_steps': 10000,  # total timesteps
                'sine_params': sine_params
            }
            domain_config = None
        else:  # domain randomization
            curriculum_config = None
            domain_config = {
                'start_point_range': ((0.35, 0.35), (0.85, 0.85)),  # Full range of possible start points
                'n_steps_range': (30, 100),  # Full range of possible steps
                'sine_params': sine_params
            }

    # Train DDPG agent
    n_parallel = 64
    model = train_ddpg(
        pretrained_model, 
        n_parallel=n_parallel, 
        total_timesteps=10000,  # Reduced timesteps for faster optimization
        curriculum_config=curriculum_config,
        domain_config=domain_config,
        task=path,
        hyperparams={
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'noise_sigma': noise_sigma,
            'buffer_size': buffer_size,
            'tau': tau
        }
    )
    
    # Evaluate the model
    eval_rewards = []
    
    if training_mode == 'single_task':
        # For single task, evaluate only on the training path
        eval_results = tracking.track_path(path, [], neural_net=model.actor, RLtrained=True)
        eval_rewards.append(np.sum(eval_results['rewards']))
    else:
        # For curriculum and domain randomization, evaluate on multiple test cases
        test_cases = [
            # Case 1: Short path, lower start point
            {
                'start_point': (0.35, 0.35),
                'sine_params': {
                    'ampSL': 0.1,
                    'ampSF': 0.1,
                    'freqSL': 2.0,
                    'freqSF': 1.0,
                    'time': 1.0,
                    'n_steps': 30
                }
            },
            # Case 2: short path, middle start point
            {
                'start_point': (0.6, 0.6),
                'sine_params': {
                    'ampSL': 0.1,
                    'ampSF': 0.1,
                    'freqSL': 2.0,
                    'freqSF': 1.0,
                    'time': 1.0,
                    'n_steps': 30
                }
            },
            # Case 3: long path, upper start point
            {
                'start_point': (0.85, 0.85),
                'sine_params': {
                    'ampSL': 0.1,
                    'ampSF': 0.1,
                    'freqSL': 2.0,
                    'freqSF': 1.0,
                    'time': 1.0,
                    'n_steps': 100
                }
            },
            # Case 4: long path, upper-lower start point
            {
                'start_point': (0.85, 0.35),
                'sine_params': {
                    'ampSL': 0.1,
                    'ampSF': 0.1,
                    'freqSL': 2.0,
                    'freqSF': 1.0,
                    'time': 1.0,
                    'n_steps': 100
                }
            },
            # Case 5: short path, lower-upper start point
            {
                'start_point': (0.35, 0.85),
                'sine_params': {
                    'ampSL': 0.1,
                    'ampSF': 0.1,
                    'freqSL': 2.0,
                    'freqSF': 1.0,
                    'time': 1.0,
                    'n_steps': 30
                }
            }
        ]
        
        for test_case in test_cases:
            eval_path, _ = tracking.generate_sinusoid_path(
                test_case['start_point'], 
                test_case['sine_params']
            )
            eval_results = tracking.track_path(eval_path, [], neural_net=model.actor, RLtrained=True)
            eval_rewards.append(np.sum(eval_results['rewards']))
    
    # Return average reward as the objective value
    return np.mean(eval_rewards)

#--------------------------------------------------

training_mode = 'single_task'
def main(training_mode=training_mode):
    """Run hyperparameter optimization.
    
    Args:
        training_mode (str): The training mode to use. One of:
            - 'single_task': Train on a single fixed path
            - 'curriculum': Use curriculum learning
            - 'domain_randomization': Use domain randomization
    """
    # Create study
    study = optuna.create_study(direction='maximize')
    
    # Run optimization
    study.optimize(lambda trial: objective(trial, training_mode), n_trials=50)  # Adjust number of trials as needed
    
    # Print results
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save results
    os.makedirs('logs/hyperparameter_search', exist_ok=True)
    study.trials_dataframe().to_csv(f'logs/hyperparameter_search/optimization_results_{training_mode}.csv')
    
    # Plot optimization history
    import optuna.visualization as vis
    fig = vis.plot_optimization_history(study)
    fig.write_image(f'logs/hyperparameter_search/optimization_history_{training_mode}.png')
    
    # Plot parameter importance
    fig = vis.plot_param_importances(study)
    fig.write_image(f'logs/hyperparameter_search/parameter_importance_{training_mode}.png')

if __name__ == "__main__":
    # You can change the training mode here
    main(training_mode=training_mode)  # Options: 'single_task', 'curriculum', 'domain_randomization' 