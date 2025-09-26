# Monte Carlo Portfolio Simulation

A simple quantitative finance system that combines technical analysis, machine learning, and Monte Carlo simulation for portfolio risk assessment and optimization.

## Project Overview

This project implements a sophisticated quantitative trading system that:
- Downloads and processes 5 years of historical stock data (2020-2024)
- Generates comprehensive technical indicators and ML features
- Trains PyTorch models to predict short-term price movements (5-day returns)
- Performs Monte Carlo simulation for portfolio risk modeling
- Provides a complete framework for quantitative trading strategies

## Data & Features

### Stock Universe
- **10 Major Stocks**: AAPL, GOOG, MSFT, AMZN, TSLA, META, NFLX, NVDA, ADBE, CSCO
- **Time Period**: 2020-2024 (5 years of historical data)
- **Data Source**: Yahoo Finance via `yfinance`

### Technical Indicators
- **Moving Averages**: 10-day and 30-day with crossover signals
- **Momentum**: 20-day and 60-day momentum calculations
- **Volatility**: 20-day rolling standard deviation
- **RSI**: 14-day Relative Strength Index
- **Trading Signals**: Moving average crossover signals
- **Target Variables**: 5-day forward returns for ML training

## Project Structure

```
├── data_handling.py      # Data download and preprocessing
├── features.py           # Technical indicator calculations
├── mc_simulation.py      # Monte Carlo portfolio simulation
├── ml_pipeline.py        # Machine learning model framework
├── backtester.py         # Backtesting functionality
├── portfolio.py          # Portfolio optimization
├── utils.py              # Utility functions
└── config.py             # Configuration settings
```

## Getting Started

### Prerequisites
```bash
pip install polars yfinance numpy matplotlib torch scikit-learn
```

### Usage

1. **Data Processing**:
   ```python
   python data_handling.py
   ```

2. **Feature Engineering**:
   ```python
   from features import features
   # Apply technical indicators to your data
   ```

3. **Monte Carlo Simulation**:
   ```python
   from mc_simulation import run_mc
   run_mc()  # Run 10,000 portfolio simulations
   ```

## 🔧 Key Components

### Data Processing (`data_handling.py`)
- Downloads stock data using `yfinance`
- Converts to Polars for high-performance data processing
- Calculates daily returns and handles missing values
- Provides clean data for downstream analysis

### Feature Engineering (`features.py`)
- Implements comprehensive technical analysis
- Generates ML-ready features and target variables
- Uses Polars for efficient rolling calculations

### Monte Carlo Simulation (`mc_simulation.py`)
- Simulates 10,000 portfolio scenarios over 252 trading days
- Uses multivariate normal distribution for return modeling
- Cholesky decomposition for correlation structure
- Visualizes portfolio value evolution

### Machine Learning Pipeline (`ml_pipeline.py`)
- **PyTorch Logistic Regression** for binary classification
- **Target**: Predicts if stock will have >1% return in next 5 days
- **Features**: All technical indicators from features.py
- **Methodology**: Time-series aware train/test split (no lookahead bias)
- **Evaluation**: Training curves, accuracy metrics, classification reports

## 📈 Monte Carlo Methodology

The simulation uses:
- **Multivariate Normal Distribution** for daily returns
- **Cholesky Decomposition** to maintain correlation structure
- **10,000 simulations** for robust statistical analysis
- **252 trading days** (1 year) projection horizon

## 🛠️ Technology Stack

- **Data Processing**: Polars (high-performance alternative to pandas)
- **Data Source**: Yahoo Finance via `yfinance`
- **Machine Learning**: PyTorch (custom neural networks)
- **Feature Engineering**: Polars (efficient rolling calculations)
- **Visualization**: Matplotlib
- **Statistical Analysis**: NumPy
- **ML Evaluation**: Scikit-learn (metrics and preprocessing)

## 📋 Current Status

- ✅ **Data Pipeline**: Complete and optimized (5 years of data)
- ✅ **Feature Engineering**: Full technical indicator suite with RSI, MA, momentum
- ✅ **ML Pipeline**: Complete PyTorch training pipeline with evaluation
- ✅ **Monte Carlo Simulation**: Fully functional risk modeling
- 🚧 **Integration**: ML predictions → Portfolio weights → MC simulation
- ⏳ **Backtesting**: Historical validation framework
- ⏳ **Portfolio Optimization**: Risk-adjusted weight optimization

## Next Steps

1. **Connect ML to Portfolio**: Convert ML predictions to portfolio weights
2. **Integrate with Monte Carlo**: Feed ML weights into risk simulation
3. **Implement Backtesting**: Historical validation of ML models
4. **Portfolio Optimization**: Risk-adjusted weight optimization algorithms
5. **Performance Metrics**: Sharpe ratio, maximum drawdown, etc.

