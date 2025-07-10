# Biped Models Package

A comprehensive Python package for simulating and analyzing bipedal walking models, with a focus on the simplest walker model and its various analytical tools.

## Overview

This package provides tools for simulating, analyzing, and optimizing bipedal walking models. The core model is the "Simplest Walker" - a passive dynamic walking model that captures the essential dynamics of human-like walking. The package includes extensive analysis capabilities for gait stability, energy efficiency, and control design.

Refer to following publication for the details of the analyses:

Mehdizadeh S, Donelan M. Controlling a simple model of bipedal walking to adapt to a wide range of target step lengths and step frequencies. bioRxiv 2025.06.23.661013; doi: https://doi.org/10.1101/2025.06.23.661013

## Installation

### Prerequisites
- Python 3.7 or higher
- Required packages (see `requirements.txt`):
  - numpy >= 1.21.0
  - scipy >= 1.7.0
  - matplotlib >= 3.5.0
  - torch >= 2.0.0 (for neural network controllers)

## Package Structure

### Core Modules

#### `biped/`
- **`BipedBaseClass.py`**: Abstract base class defining the interface for all bipedal models. Provides common methods for simulation, analysis, and control.

#### `models/simplest_walker/`
- **`SimplestWalker.py`**: Main implementation of the simplest walker model. Includes:
  - Single step simulation with foot contact detection
  - Energy calculation (push-off, collision, swing work)
  - Linear analysis capabilities
  - Animation and visualization tools
  - Feedback control implementation

### Analysis Modules (`models/simplest_walker/analysis/`)

#### Core Analysis
- **`simulate_walker.py`**: Multi-step simulation capabilities for the walker model
- **`linear_analysis.py`**: Linearization of dynamics around nominal trajectories, calculation of A, B, C, D matrices
- **`analyze_gait_stability.py`**: Stability analysis using eigenvalues and eigenvectors

#### Energy Analysis
- **`cost_of_transport.py`**: Comprehensive energy analysis across the gait space, including:
  - Push-off work calculation
  - Collision energy analysis
  - Swing leg work computation
  - Total cost of transport mapping
  - 3D visualization of energy landscapes

#### Control Analysis
- **`optimal_step_response.py`**: Analysis of optimal step responses and control performance
- **`tracking_analysis.py`**: Tracking performance analysis for different control strategies
- **`maximum_tolerable_perturbation.py`**: Robustness analysis by finding maximum tolerable perturbations
- **`NeuralNetworkController.py`**: Neural network-based controller implementation

#### Utilities
- **`utils.py`**: Common utilities for data processing, loading, and analysis functions

### Optimization Modules (`models/simplest_walker/optimization/`)

- **`find_gait.py`**: Optimization tools for finding limit cycle solutions:
  - Periodicity constraints
  - Step length and frequency constraints

### Visualization Modules (`models/simplest_walker/plotting/`)

- **`plot_functions.py`**: Comprehensive plotting utilities:
  - Eigenvalue contour plots
  - Control gain visualizations
  - State and control matrix plots
  - Energy landscape visualizations
  - Stability region plots

### Scripts (`scripts/simplest_walker/`)

#### Simulation Scripts
- **`run_simulation.py`**: Command-line interface for running walker simulations
- **`run_stability_analysis.py`**: Stability analysis execution script

#### Analysis Scripts
- **`run_linear_analysis.py`**: Linear dynamics analysis execution
- **`run_cost_of_transport.py`**: Energy analysis execution
- **`run_max_tol_pert.py`**: Maximum tolerable perturbation analysis
- **`run_tracking_analysis.py`**: Tracking performance analysis
- **`run_optimal_step_response.py`**: Optimal step response analysis

#### Optimization Scripts
- **`run_find_gait.py`**: Gait optimization execution
- **`find_optimal_step_response.py`**: Optimal step response finding
- **`hyperparameter_search.py`**: Hyperparameter optimization tools

#### Visualization Scripts
- **`run_plotting.py`**: Plot generation and visualization

#### Training Scripts
- **`train_NN.py`**: Neural network controller training

## Data Structure

The package uses several data files stored in `data/simplest_walker/`:
- `limit_cycle_solutions.npz`: Pre-computed limit cycle solutions
- `linear_analysis_results.npz`: Linear analysis results
- `maximum_tolerable_perturbation_results.npz`: Robustness analysis results

## License

MIT License

Copyright (c) 2024 Sina Mehdizadeh, Max Donelan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Disclaimer

**IMPORTANT DISCLAIMER**: This software is provided "as is" without warranty of any kind. The authors and contributors make no representations or warranties about the accuracy, completeness, reliability, suitability, or availability of the software or the information, products, services, or related graphics contained in the software for any purpose. Any reliance you place on such information is therefore strictly at your own risk.

This package is intended for research and educational purposes. The models and analyses provided are simplified representations of complex biomechanical systems and should not be used for clinical or medical applications without proper validation and expert review. The authors are not responsible for any decisions made based on the results obtained from this software.

Users should:
- Validate all results against experimental data when possible
- Understand the limitations of the simplified models
- Consult with domain experts for critical applications
- Use appropriate safety measures when applying results to real systems

## Citation

If you use this package in your research, please cite:

Mehdizadeh S, Donelan M. Controlling a simple model of bipedal walking to adapt to a wide range of target step lengths and step frequencies. bioRxiv 2025.06.23.661013; doi: https://doi.org/10.1101/2025.06.23.661013

## Contact

For questions, issues, or contributions, please contact:
Sina Mehdizadeh: drsinamehdizadeh@gmail.com
Max Donelan: mdonelan@sfu.ca 

