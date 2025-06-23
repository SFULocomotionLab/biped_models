import numpy as np

def load_limit_cycle_solutions(file_path='data/simplest_walker/limit_cycle_solutions.npz'):
    """
    Load limit cycle solutions from a .npz file.
    
    Args:
        file_path (str): Path to the .npz file containing limit cycle solutions
        
    Returns:
        tuple: (solutions, target_step_length, target_step_frequency)
    """
    try:
        loaded_data = np.load(file_path, allow_pickle=True)
        solutions = loaded_data['arr_0']
        
        # Define the target ranges
        target_step_length = np.linspace(1.1, 0.1, 101)
        target_step_frequency = np.linspace(0.1, 1.1, 101)
        
        return solutions, target_step_length, target_step_frequency
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find limit cycle solutions file at {file_path}")
    except KeyError:
        raise KeyError("The .npz file does not contain the expected 'arr_0' key")
    except Exception as e:
        raise Exception(f"Error loading limit cycle solutions: {str(e)}")

def load_linear_analysis_data(file_path='data/simplest_walker/linear_analysis_results.npz'):
    """
    Load data from linear analysis results.
    
    Args:
        file_path (str): Path to the .npz file containing linear analysis results
        
    Returns:
        dict: Loaded linear analysis data
    """
    try:
        return np.load(file_path, allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find linear analysis results file at {file_path}")
    except Exception as e:
        raise Exception(f"Error loading linear analysis results: {str(e)}") 