# Monte Carlo Portfolio Simulation

A quantitative finance project that combines machine learning with Monte Carlo simulation for portfolio risk assessment and optimization.

## 🎯 Project Overview

This project implements a sophisticated quantitative trading system that:
- Downloads and processes historical stock data
- Generates technical indicators and features
- Uses machine learning to predict future returns
- Performs Monte Carlo simulation for portfolio risk modeling

## 📊 Data & Features

### Stock Universe
- **10 Major Stocks**: AAPL, GOOG, MSFT, AMZN, TSLA, META, NFLX, NVDA, ADBE, CSCO
- **Time Period**: 2020-2024 (5 years of historical data)
- **Data Source**: Yahoo Finance via `yfinance`

### Technical Indicators
- **Moving Averages**: 10-day and 30-day
- **Momentum**: 20-day and 60-day momentum
- **Volatility**: 20-day rolling standard deviation
- **RSI**: 14-day Relative Strength Index
- **Trading Signals**: Moving average crossover signals

## 🏗️ Project Structure

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

## 🚀 Getting Started

### Prerequisites
```bash
pip install polars yfinance numpy matplotlib
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
- Framework for multiple ML models (XGBoost, LightGBM, CatBoost, Neural Networks)
- Ready for integration with feature engineering
- Will generate portfolio weights for Monte Carlo simulation

## 📈 Monte Carlo Methodology

The simulation uses:
- **Multivariate Normal Distribution** for daily returns
- **Cholesky Decomposition** to maintain correlation structure
- **10,000 simulations** for robust statistical analysis
- **252 trading days** (1 year) projection horizon

## 🛠️ Technology Stack

- **Data Processing**: Polars (high-performance alternative to pandas)
- **Data Source**: Yahoo Finance
- **Machine Learning**: XGBoost, LightGBM, CatBoost, TensorFlow
- **Visualization**: Matplotlib
- **Statistical Analysis**: NumPy

## 📋 Current Status

- ✅ **Data Pipeline**: Complete and optimized
- ✅ **Feature Engineering**: Full technical indicator suite
- ✅ **Monte Carlo Simulation**: Fully functional
- 🚧 **ML Pipeline**: Framework ready for implementation
- ⏳ **Integration**: ML weights → Portfolio optimization

## 🎯 Next Steps

1. Complete machine learning model training
2. Generate portfolio weights from ML predictions
3. Integrate ML weights into Monte Carlo simulation
4. Implement backtesting for model validation
5. Add portfolio optimization algorithms

## 📝 Notes

- Uses Polars instead of pandas for better performance
- Forward-fills missing values to avoid lookahead bias
- Equal weights currently used (waiting for ML optimization)
- Designed for quantitative finance applications

## 🤝 Contributing

This is a quantitative finance research project focused on portfolio optimization and risk management.
