import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from config import SIMULATION_CONFIG, PLOT_CONFIG, RISK_CONFIG

def load_portfolio_weights(csv_path : str) -> Tuple[np.ndarray, list]:
    df = pl.read_csv(csv_path)
    weights = df['portfolio_weight'].to_numpy()
    symbols = df['symbol'].to_list()

    print('Loaded weights.')
    print('portfolio composition: ')
    for symbol, weight in zip(symbols, weights):
        print(f'{symbol}: {weight:.2%}')
    
    return weights, symbols

def get_return_statistics(returns_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean_returns = np.mean(returns_data, axis=0)
    cov_matrix = np.cov(returns_data.T)

    return mean_returns, cov_matrix

def simulate_portfolio_paths(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray
) -> np.ndarray:
    n_sims = SIMULATION_CONFIG['n_sims']
    n_days = SIMULATION_CONFIG['n_days']
    initial_investment = SIMULATION_CONFIG['initial_investment']

    print('Running Monte Carlo Simulation')
    print(f'{n_sims} simulations of {n_days} days, with initial investment of {initial_investment}')

    n_stocks = len(weights)

    # Initialise portfolio simulations array
    portfolio_sims = np.zeros((n_days, n_sims))

    # Cholesky decomposition of correlated random returns
    L = np.linalg.cholesky(cov_matrix)

    for sim in range(n_sims):
        # Generate correlated random returns using Cholesky decomposition
        Z = np.random.normal(size=(n_days, n_stocks))
        daily_returns = mean_returns + Z @ L.T

        # Calculate weighted portfolio returns
        portfolio_returns = daily_returns @ weights

        # Calculate cumulative portfolio value
        portfolio_sims[:, sim] = initial_investment * np.cumprod(1 + portfolio_returns)
    
    print('Simulations completed')

    return portfolio_sims

def calculate_sim_stats(portfolio_sims: np.ndarray) -> dict:
    final_values = portfolio_sims[-1, :]  # Final portfolio values for each simulation
    percentiles = RISK_CONFIG['percentiles']

    stats = {
        'mean_final_value': np.mean(final_values),
        'median_final_value': np.median(final_values),
        'min_final_value': np.min(final_values),
        'max_final_value': np.max(final_values),
        'std_final_value': np.std(final_values),
        'percentile_5': np.percentile(final_values, percentiles[0]),
        'percentile_25': np.percentile(final_values, percentiles[1]),
        'percentile_75': np.percentile(final_values, percentiles[2]),
        'percentile_95': np.percentile(final_values, percentiles[3]),
    }

    return stats

def plot_sim_paths(portfolio_sims: np.ndarray):
    n_paths_to_plot = PLOT_CONFIG['n_paths_to_plot']
    initial_investment = SIMULATION_CONFIG['initial_investment']
    figure_size = PLOT_CONFIG['figure_size_paths']
    alpha = PLOT_CONFIG['alpha_transparency']
    grid_alpha = PLOT_CONFIG['grid_alpha']
    
    n_days = portfolio_sims.shape[0]
    plt.figure(figsize=figure_size)

    # Plot paths
    sample_indices = np.random.choice(n_days, size=n_paths_to_plot, replace=False)
    plt.plot(sample_indices, portfolio_sims[sample_indices, :].T, alpha=alpha, color='blue')

    # Plot mean path
    mean_path = np.mean(portfolio_sims, axis=1)
    plt.plot(mean_path, 'r-', linewidth=2, label='Mean Path')

    # Plot initial investment line
    plt.axhline(y=initial_investment, color='black', linestyle='--', 
                linewidth=1, label='Initial Investment')
    
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Trading Days')
    plt.title(f'Monte Carlo Simulation: Portfolio Value Paths (showing {len(sample_indices)} of {portfolio_sims.shape[1]})')
    plt.legend()
    plt.grid(True, alpha=grid_alpha)
    plt.tight_layout()
    plt.show()

def plot_final_value_distribution(portfolio_sims: np.ndarray):
    initial_investment = SIMULATION_CONFIG['initial_investment']
    figure_size = PLOT_CONFIG['figure_size_histogram']
    bins = PLOT_CONFIG['histogram_bins']
    grid_alpha = PLOT_CONFIG['grid_alpha']
    
    final_values = portfolio_sims[-1, :]

    plt.figure(figsize=figure_size)
    plt.hist(final_values, bins=bins, density=True, alpha=0.7, color='blue')
    
    # Add vertical lines for key statistics
    plt.axvline(x=np.mean(final_values), color='red', linestyle='--', linewidth=2, label='Mean')
    plt.axvline(x=np.median(final_values), color='green', linestyle='--', linewidth=2, label='Median')
    plt.axvline(x=initial_investment, color='black', linestyle='--', linewidth=2, label='Initial Investment')

    # Add percentile lines
    p5 = np.percentile(final_values, 5)
    p25 = np.percentile(final_values, 25)
    p75 = np.percentile(final_values, 75)
    p95 = np.percentile(final_values, 95)
    plt.axvline(x=p5, color='orange', linestyle='--', linewidth=2, label='5th Percentile')
    plt.axvline(x=p25, color='yellow', linestyle='--', linewidth=2, label='25th Percentile')
    plt.axvline(x=p75, color='yellow', linestyle='--', linewidth=2, label='75th Percentile')
    plt.axvline(x=p95, color='orange', linestyle='--', linewidth=2, label='95th Percentile')
    
    plt.xlabel('Final Portfolio Value ($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Final Portfolio Values')
    plt.legend()
    plt.grid(True, alpha=grid_alpha)
    plt.tight_layout()
    plt.show()

def print_simulation_summary(stats: dict):
    initial_investment = SIMULATION_CONFIG['initial_investment']
    # Print simulation summary
    print("\n" + "=" * 60)
    print("MONTE CARLO SIMULATION SUMMARY")
    print("=" * 60)
    
    print(f"\nInitial Investment: ${initial_investment:,.0f}")
    print(f"\nFinal Portfolio Value Statistics:")
    print(f"  Mean:                ${stats['mean_final_value']:,.0f}")
    print(f"  Median:              ${stats['median_final_value']:,.0f}")
    print(f"  Std Dev:             ${stats['std_final_value']:,.0f}")
    print(f"  Min:                 ${stats['min_final_value']:,.0f}")
    print(f"  Max:                 ${stats['max_final_value']:,.0f}")
    
    print(f"\nPercentiles:")
    print(f"  5th percentile:      ${stats['percentile_5']:,.0f}")
    print(f"  25th percentile:     ${stats['percentile_25']:,.0f}")
    print(f"  75th percentile:     ${stats['percentile_75']:,.0f}")
    print(f"  95th percentile:     ${stats['percentile_95']:,.0f}")
    
    expected_return = (stats['mean_final_value'] - initial_investment) / initial_investment * 100
    print(f"\nExpected Return:       {expected_return:.2f}%")
    
    print("=" * 60)

def run_mc_simulation(
    weights: Optional[np.ndarray] = None,
    csv_path: Optional[str] = None,
    returns_data: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, dict]:
    """
    Main function to run Monte Carlo simulation
    
    Args:
        weights: Portfolio weights (optional if csv_path provided)
        csv_path: Path to CSV file with weights (optional if weights provided)
        returns_data: Historical returns data
        
    Returns:
        portfolio_sims: Array of simulated portfolio values
        stats: Dictionary of simulation statistics
    """
    # Load weights if CSV path provided
    if csv_path is not None:
        weights, symbols = load_portfolio_weights(csv_path)
    elif weights is None:
        raise ValueError("Must provide either weights or csv_path")
    
    # Get returns data if not provided
    if returns_data is None:
        from data_handling import returns_pl
        returns_data = returns_pl.to_numpy()
    
    # Calculate return statistics
    mean_returns, cov_matrix = get_return_statistics(returns_data)
    
    # Run simulation
    portfolio_sims = simulate_portfolio_paths(
        weights=weights,
        mean_returns=mean_returns,
        cov_matrix=cov_matrix
    )
    
    # Calculate statistics
    stats = calculate_sim_stats(portfolio_sims)
    
    # Print summary
    print_simulation_summary(stats)
    
    # Generate plots
    plot_sim_paths(portfolio_sims)
    plot_final_value_distribution(portfolio_sims)
    
    return portfolio_sims, stats


# Example usage
if __name__ == "__main__":
    # Example 1: Using equal weights from data_handling
    from data_handling import stock_list, returns_pl
    
    returns_data = returns_pl.to_numpy()
    equal_weights = np.ones(len(stock_list)) / len(stock_list)
    
    portfolio_sims, stats = run_mc_simulation(
        weights=equal_weights,
        returns_data=returns_data
    )
    
    # Example 2: Using weights from CSV (uncomment when you have the file)
    # portfolio_sims, stats = run_mc_simulation(
    #     csv_path='portfolio_weights_20250101_120000.csv'
    # )