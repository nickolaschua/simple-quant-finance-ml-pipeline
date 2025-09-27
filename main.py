import yfinance as yf
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Custom modules
from features import (
    calculate_ma, 
    calculate_momentum, 
    calculate_volatility, 
    calculate_rsi
)
from ml_pipeline import run_ml_pipeline

def collect_data(stock_list, start_date='2020-01-01', end_date='2024-12-31'):
    """
    Data collection step - download stock data and calculate returns
    """
    print("=" * 50)
    print("STEP 1: DATA COLLECTION")
    print("=" * 50)
    
    print(f"Downloading data for {len(stock_list)} stocks...")
    print(f"Date range: {start_date} to {end_date}")
    
    # Download data using yfinance
    data = yf.download(stock_list, start=start_date, end=end_date)
    data = data['Close']
    data = data.ffill()  # Forward fill missing values
    
    # Convert to long format for Polars
    data_reset = data.reset_index()
    df = pl.from_pandas(data_reset)
    
    # Reshape to long format (one row per stock-date combination)
    df_long = df.melt(
        id_vars=['Date'], 
        variable_name='symbol', 
        value_name='close'
    ).with_columns([
        pl.col('Date').alias('date'),
        pl.col('close').pct_change().over('symbol').alias('returns')
    ]).drop_nulls()  # Remove first day (NaN returns)
    
    print(f"Data collected: {df_long.shape[0]} rows, {df_long['symbol'].n_unique()} stocks")
    print(f"Date range: {df_long['date'].min()} to {df_long['date'].max()}")
    print("\nStock symbols:", df_long['symbol'].unique().to_list())
    
    return df_long

def engineer_features(df):
    """
    Feature engineering step - create technical indicators
    """
    print("\n" + "=" * 50)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 50)
    
    print("Creating technical indicators for each stock...")
    
    # Apply feature engineering to each stock separately
    features_df = (df
        .group_by('symbol')
        .map_groups(lambda group: 
            group
            .pipe(calculate_ma, 'close')  # MA_10, MA_30
            .pipe(calculate_momentum, 'close')  # momentum_20d, momentum_60d
            .pipe(calculate_volatility, 'returns')  # volatility_20d
            .pipe(calculate_rsi, 'close')  # rsi_14d
            .with_columns([
                # MA signal: 1 if short MA > long MA, 0 otherwise
                (pl.col('ma_10') > pl.col('ma_30')).cast(pl.Int8).alias('ma_signal')
            ])
        )
        .drop_nulls()  # Remove rows with NaN from rolling calculations
    )
    
    print(f"Features created: {features_df.shape[0]} rows")
    print("Feature columns:", [col for col in features_df.columns if col not in ['date', 'symbol', 'close', 'returns']])
    
    # Show feature summary by stock
    feature_counts = features_df.group_by('symbol').count().sort('symbol')
    print("\nRows per stock after feature engineering:")
    for row in feature_counts.iter_rows():
        print(f"  {row[0]}: {row[1]} rows")
    
    return features_df

def run_ml_training(df):
    """
    ML training step - train logistic regression model
    """
    print("\n" + "=" * 50)
    print("STEP 3: MACHINE LEARNING PIPELINE")
    print("=" * 50)
    
    # Define feature columns (exclude non-feature columns)
    feature_cols = [
        'ma_10', 'ma_30', 'ma_signal', 
        'momentum_20d', 'momentum_60d', 
        'volatility_20d', 'rsi_14d'
    ]
    
    print(f"Using features: {feature_cols}")
    print(f"Target: Binary classification (1 = positive returns, 0 = negative/neutral)")
    
    # Run ML pipeline
    model, scaler, predictions_proba, test_df = run_ml_pipeline(
        df, 
        feature_cols, 
        target_threshold=0.01,  # 1% return threshold
        forward_days=5  # Predict 5-day forward returns
    )
    
    return model, scaler, predictions_proba, test_df, feature_cols

def analyze_predictions(predictions_proba, test_df):
    """
    Analyze model predictions and generate portfolio insights
    """
    print("\n" + "=" * 50)
    print("STEP 4: PREDICTION ANALYSIS")
    print("=" * 50)
    
    # Add predictions back to test dataframe
    test_with_preds = test_df.with_columns([
        pl.Series('prediction_proba', predictions_proba)
    ])
    
    # Analyze predictions by stock
    print("Prediction summary by stock:")
    pred_summary = (test_with_preds
        .group_by('symbol')
        .agg([
            pl.col('prediction_proba').count().alias('n_predictions'),
            pl.col('prediction_proba').mean().alias('avg_probability'),
            pl.col('target').mean().alias('actual_positive_rate'),
            (pl.col('prediction_proba') > 0.5).sum().alias('predicted_positive')
        ])
        .sort('avg_probability', descending=True)
    )
    
    for row in pred_summary.iter_rows(named=True):
        print(f"  {row['symbol']}: Avg Prob={row['avg_probability']:.3f}, "
              f"Actual Positive Rate={row['actual_positive_rate']:.3f}, "
              f"Predicted Positive={row['predicted_positive']}/{row['n_predictions']}")
    
    # Generate portfolio weights based on predictions
    print("\nGenerating portfolio weights...")
    
    # Get latest prediction for each stock (for portfolio construction)
    latest_predictions = (test_with_preds
        .group_by('symbol')
        .agg([
            pl.col('prediction_proba').last().alias('latest_probability'),
            pl.col('date').last().alias('latest_date')
        ])
        .with_columns([
            # Simple weight scheme: normalize probabilities to sum to 1
            (pl.col('latest_probability') / pl.col('latest_probability').sum()).alias('portfolio_weight')
        ])
        .sort('portfolio_weight', descending=True)
    )
    
    print("\nSuggested Portfolio Weights (based on latest predictions):")
    total_weight = 0
    for row in latest_predictions.iter_rows(named=True):
        weight_pct = row['portfolio_weight'] * 100
        total_weight += weight_pct
        print(f"  {row['symbol']}: {weight_pct:.1f}% (prob={row['latest_probability']:.3f})")
    print(f"  Total: {total_weight:.1f}%")
    
    return latest_predictions

def save_results(model, scaler, predictions_df, feature_cols):
    """
    Save key outputs for later use
    """
    print("\n" + "=" * 50)
    print("STEP 5: SAVING RESULTS")
    print("=" * 50)
    
    import torch
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model and scaler
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_cols': feature_cols,
        'timestamp': timestamp
    }, f'trained_model_{timestamp}.pth')
    
    # Save scaler
    import joblib
    joblib.dump(scaler, f'feature_scaler_{timestamp}.pkl')
    
    # Save predictions
    predictions_df.write_csv(f'portfolio_weights_{timestamp}.csv')
    
    print(f"Saved:")
    print(f"  - Model: trained_model_{timestamp}.pth")
    print(f"  - Scaler: feature_scaler_{timestamp}.pkl") 
    print(f"  - Portfolio weights: portfolio_weights_{timestamp}.csv")

def main():
    """
    Main orchestration function - runs the complete pipeline
    """
    print("QUANTITATIVE TRADING PIPELINE STARTED")
    print("=" * 60)
    
    # Configuration
    stock_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'ADBE', 'CSCO']
    start_date = '2020-01-01'
    end_date = '2024-12-31'
    
    try:
        # Step 1: Data Collection
        raw_data = collect_data(stock_list, start_date, end_date)
        
        # Step 2: Feature Engineering  
        features_data = engineer_features(raw_data)
        
        # Step 3: ML Training
        model, scaler, predictions, test_data, feature_cols = run_ml_training(features_data)
        
        # Step 4: Prediction Analysis
        portfolio_weights = analyze_predictions(predictions, test_data)
        
        # Step 5: Save Results
        save_results(model, scaler, portfolio_weights, feature_cols)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Review the portfolio weights above")
        print("2. Run Monte Carlo simulation (mc.py) with these weights")
        print("3. Backtest the strategy performance")
        
        return model, scaler, portfolio_weights, feature_cols
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()