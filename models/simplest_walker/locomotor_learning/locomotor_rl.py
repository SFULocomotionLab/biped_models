import numpy as np

class LocomotorRL:
    """
    Implementation of the reinforcement learning framework for locomotor adaptation
    as described in "Exploration-based learning of a stabilizing controller predicts locomotor adaptation"
    (Nature Communications, 2024).
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 controller_params: np.ndarray,
                 learning_rate: float = 0.01,
                 memory_learning_rate: float = 0.001,
                 memory_formation_rate: float = 0.001
                 ):
        """
        Initialize the locomotor RL framework.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            controller_params_dim: Dimension of the controller parameters
            learning_rate: Learning rate for parameter updates
            exploration_noise: Standard deviation of exploration noise
            memory_size: Size of the memory buffer for storing experiences
            memory_learning_rate: Learning rate for memory parameter updates
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.controller_params = controller_params
        self.learning_rate = learning_rate
        self.memory_learning_rate = memory_learning_rate
        self.memory_formation_rate = memory_formation_rate
        
        # Initialize memory functions Fp(λ) and FJ(λ)
        # For simplicity, we use linear functions with slopes and intercepts
        controller_params_dim = self.controller_params.shape[0]
        self.memory_params = {
            'Fp_slope': np.zeros((controller_params_dim, 1)),  # Slope for controller parameters
            'Fp_intercept': np.zeros((controller_params_dim, 1)),  # Intercept for controller parameters
            'FJ_slope': np.zeros((1, 1)),  # Slope for performance
            'FJ_intercept': np.zeros((1, 1))  # Intercept for performance
        }

    def fit_J_linear_model(self, states: np.ndarray, controller_params: np.ndarray, J: np.ndarray):
        """
        Fit a linear model J = F*s + G*p using least squares regression. (eq. 5)
        
        Args:
            states: Tensor of states (s)
            controller_params: Tensor of controller parameters (p) 
            J: Tensor of performance values
            
        Returns:
            F: State coefficient matrix
            G: Parameter coefficient matrix
        """
        # Combine states and parameters into design matrix X
        X = np.concatenate([states, controller_params], axis=1)
        
        # Solve least squares: J = X * [F; G]
        coefficients, _ = np.linalg.lstsq(X, J)
        
        # Split coefficients into F and G matrices
        F = coefficients[:self.state_dim, :]
        G = coefficients[self.state_dim:, :]
        
        return F, G
        
    def fit_internal_model(self, states: np.ndarray, actions: np.ndarray, 
                           controller_params: np.ndarray, next_states: np.ndarray):
        """
        Fit the internal model using least squares regression. (eq. 6)
        s(i+1) = A*s(i) + B*p(i)
        """
        # Combine states and parameters into design matrix X
        X = np.concatenate([states, controller_params], axis=1)
        
        # Solve least squares: J = X * [F; G]
        coefficients, _ = np.linalg.lstsq(X, next_states)
        
        # Split coefficients into F and G matrices
        A = coefficients[:self.state_dim, :]
        B = coefficients[self.state_dim:, :]
        
        return A, B
    
    def compute_gradient(self, A: np.ndarray, B: np.ndarray, F: np.ndarray, G: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the performance objective with respect to controller parameters (Eq. 7).
        
        Args:
           A, B, F, G: Coefficients of the linear models
            
        Returns:
            Gradient of J with respect to controller parameters
        """
        I = np.eye(A.shape[0])
        gradient_J = G + F@ np.linalg.inv(I - A) @ B
        
        return gradient_J
    
    def update_parameters(self, gradient: np.ndarray):
        """
        Update controller parameters using the computed gradient.
        
        Args:
            gradient: Computed gradient of J
        """

        gradient_term = - self.learning_rate * gradient
        memory_term = self.memory_learning_rate * \
            (self.get_controller_params_from_memory(self.task_params)-self.controller_params)
        
        self.controller_params = self.controller_params + gradient_term + memory_term

    def get_controller_params_from_memory(self, task_params: np.ndarray) -> np.ndarray:
        """
        Get controller parameters from memory for given task parameters.
        Fp(λ) = slope * λ + intercept
        
        Args:
            task_params: Task parameters (e.g., walking speed)
            
        Returns:
            Controller parameters from memory
        """
        return self.memory_params['Fp_slope'] @ task_params + self.memory_params['Fp_intercept']
    
    def get_performance_from_memory(self, task_params: np.ndarray) -> np.ndarray:
        """
        Get performance value from memory for given task parameters.
        FJ(λ) = slope * λ + intercept
        
        Args:
            task_params: Task parameters (e.g., walking speed)
            
        Returns:
            Performance value from memory
        """
        return self.memory_params['FJ_slope'] @ task_params + self.memory_params['FJ_intercept']
    
    def update_memory(self, task_params: np.ndarray, current_controller: np.ndarray, 
                     current_performance: np.ndarray):
        """
        Update memory functions when current performance is better than stored memory.
        The loss L measures how well the memory approximates the current controller.
        
        Args:
            task_params: Task parameters
            current_controller: Current controller parameters
            current_performance: Current performance value
        """
        # Get memory predictions
        controller_params_from_memory = self.get_controller_params_from_memory(task_params)
        performance_from_memory = self.get_performance_from_memory(task_params)
        
        # Update memory if current performance is better than stored memory
        if current_performance < performance_from_memory:
            # Compute loss L as RMSE over controller parameters
            # L = (controller_params_from_memory - current_controller) ** 2
            
            # Compute gradients of L with respect to memory parameters
            # For Fp parameters (controller memory)
            Fp_grad_slope = (controller_params_from_memory - current_controller) @ task_params.T
            Fp_grad_intercept = controller_params_from_memory - current_controller
            
            # For FJ parameters (performance memory)
            FJ_grad_slope = (performance_from_memory - current_performance) @ task_params.T
            FJ_grad_intercept = performance_from_memory - current_performance
            
            # Update memory parameters using gradient descent
            self.memory_params['Fp_slope'] -= self.memory_formation_rate * Fp_grad_slope
            self.memory_params['Fp_intercept'] -= self.memory_formation_rate * Fp_grad_intercept
            self.memory_params['FJ_slope'] -= self.memory_formation_rate * FJ_grad_slope
            self.memory_params['FJ_intercept'] -= self.memory_formation_rate * FJ_grad_intercept