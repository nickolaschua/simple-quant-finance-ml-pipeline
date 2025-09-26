# Defining the investment universe
import yfinance as yf
import polars as pl
import numpy as np

# To keep it simple, we will use the the top 10 stocks in the S&P 500
stock_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'ADBE', 'CSCO']

# Downloading all the data
data = yf.download(stock_list, start='2020-01-01', end='2024-12-31')

# Get the Close prices (not Adj Close)
data = data['Close']

# Convert pandas DataFrame to Polars DataFrame
data_pl = pl.from_pandas(data)

# Forward filling the missing values
data_pl = data_pl.fill_null(strategy="forward")

# Calculating the returns
returns_pl = data_pl.select([
    pl.col(col).pct_change().alias(col) for col in data_pl.columns
])

# Drop the first row (NaN values) from returns
returns_pl = returns_pl.drop_nulls()

# Quick data check
print("Data shape:", data_pl.shape)
print("Returns shape:", returns_pl.shape)
print("\nMissing values in prices:", data_pl.null_count().sum_horizontal().item())
print("Missing values in returns:", returns_pl.null_count().sum_horizontal().item())

