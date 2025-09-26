# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np

# Import data from data_handling module
def get_data():
    # Use the already processed data from data_handling.py
    from data_handling import data_pl, returns_pl
    
    # Convert to numpy for mean and covariance calculations
    returns_np = returns_pl.to_numpy()
    mean_returns = np.mean(returns_np, axis=0)
    cov_matrix = np.cov(returns_np.T)
    
    return mean_returns, cov_matrix

# Importing the stock list
from data_handling import stock_list

# Basic MC Simulation
def run_mc():
    # Use the data from data_handling.py (no need to download again)
    mean_returns, cov_matrix = get_data()

    # Equal weights for now - replace with ML algorithm weights later
    weights = np.array([1.0] * len(stock_list)) / len(stock_list)

    # Monte Carlo Simulation

    n_simulations = 10000 # number of simulations
    n_days = 252 # timeframe in days (should i use number of trading days or just days)

    mean_matrix = np.full(shape=(n_days, n_simulations), fill_value=mean_returns)
    mean_matrix = mean_matrix.T

    portfolio_sims = np.full(shape=(n_days, n_simulations), fill_value=0.0)

    initial_investment = 100000

    for matrix in range(0, n_simulations):
        # Monte Carlo Loop
        # We will be assuming daily returns are distributed by a MultiVariate Normal Distribution
        Z = np.random.normal(size=(n_days, len(weights)))
        L = np.linalg.cholesky(cov_matrix) # Cholseky Decomposition
        daily_returns = mean_matrix + np.inner(L, Z)
        portfolio_sims[:, matrix] = np.cumprod(np.inner(weights, daily_returns.T) + 1) * initial_investment

    plt.plot(portfolio_sims)
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Days')
    plt.title('Monte Carlo Simulation of Portfolio Value')
    plt.show()