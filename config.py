"""
Monte Carlo Simulation Configuration
All simulation parameters are centralized here for easy modification.
"""

# Core Simulation Parameters
SIMULATION_CONFIG = {
    'n_sims': 10000,
    'n_days': 252,
    'initial_investment': 100000,
}

# Plotting and Visualization Parameters
PLOT_CONFIG = {
    'n_paths_to_plot': 100,  # Number of paths to show in simulation plots
    'figure_size_paths': (12, 6),  # Figure size for simulation paths plot
    'figure_size_histogram': (10, 6),  # Figure size for histogram plot
    'histogram_bins': 50,  # Number of bins for histogram
    'alpha_transparency': 0.1,  # Transparency for simulation paths
    'grid_alpha': 0.3,  # Grid transparency
}

# Risk Analysis Parameters
RISK_CONFIG = {
    'percentiles': [5, 25, 75, 95],  # Percentiles to calculate
}

# Portfolio Parameters
PORTFOLIO_CONFIG = {
    'default_weights_file': 'portfolio_weights.csv',
    'equal_weight_strategy': True,  # Use equal weights if no CSV provided
}
