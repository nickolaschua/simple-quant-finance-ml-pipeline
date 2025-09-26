import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out

def prepare_data(df, target_threshold=0.01, forward_days=5):
    # Create binary target variable
    ml_df = df.with_columns([
        # Forward returns
        pl.col('returns').shift(-forward_days).alias('forward_returns'),

    ]).with_columns([
        # Binary target: 1 if forward_returns > target_threshold, 0 otherwise
        (pl.col('forward_returns') > target_threshold).cast(pl.Int32).alias('target')
    ]).drop_nulls()
    
    return ml_df
    
# Split data into training and testing sets while ensuring no lookahead bias
def train_test_split_by_date(df, train_ratio=0.8):
    # Sort by date to ensure no lookahead bias
    df_sorted = df.sort('date')

    # Calculate split index
    total_days = df_sorted.height[0]
    split_idx = int(total_days * train_ratio)

    # Get the split date
    split_date = df_sorted.row(split_idx)['date']

    # Split data
    train_df = df_sorted.filter(pl.col('date') < split_date)
    test_df = df_sorted.filter(pl.col('date') >= split_date)
    
    return train_df, test_df

def prepare_X_y(df, feature_cols):
    # X: features, y: target
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y = df['target'].to_numpy().flatten().astype(np.float32)

    return X, y

def train_model(X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=100):
    # Get input_size from X_train
    input_size = X_train.shape[1]

    # Initialize model, with loss function and optimizer
    model = LogisticRegression(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)

    # Training loop
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    return model, train_losses, val_losses


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        predictions = model(X_test_tensor)
        predictions_binary = (predictions > 0.5).float().numpy().flatten()

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions_binary)
    print(f'Accuracy: {accuracy:.4f}')
    print(classification_report(y_test, predictions_binary))

    return predictions.numpy().flatten(), predictions_binary

def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def run_ml_pipeline(df, feature_cols, target_threshold=0.01, forward_days=5):
    # Prepare data
    ml_df = prepare_data(df, target_threshold, forward_days)

    # Split data
    train_df, test_df = train_test_split_by_date(ml_df, train_ratio=0.8)

    # Prepare features and targets
    X_train, y_train = prepare_X_y(train_df, feature_cols)
    X_test, y_test = prepare_X_y(test_df, feature_cols)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model

    val_size = int(len(X_train_scaled) * 0.2)
    X_val = X_train_scaled[-val_size:]
    y_val = y_train[-val_size:]
    X_train_final = X_train_scaled[:-val_size]
    y_train_final = y_train[:-val_size]

    model, train_losses, val_losses = train_model(X_train_final, y_train_final, X_val, y_val)

    # Evaluate model
    predictions_proba, predictions_binary = evaluate_model(model, X_test_scaled, y_test)

    # Plot training curves
    plot_training_curves(train_losses, val_losses)

    return model, scaler, predictions_proba, test_df
    