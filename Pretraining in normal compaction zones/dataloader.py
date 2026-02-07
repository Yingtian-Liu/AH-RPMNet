"""
Data loading and preprocessing functions.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import DATA_PATH, RATIO, FACTOR, TRAIN_START, TRAIN_END, FEATURE_COLS, TRAIN_CONFIG

def load_and_preprocess_data():
    """
    Load CSV data and preprocess for training.
    Returns:
        x_train, y_train, x_test, y_test, depth_train, depth_test
    """
    # Load data
    df = pd.read_csv(DATA_PATH).values
    
    # Extract features with downsampling
    depth = df[:, FEATURE_COLS['depth']][::RATIO, np.newaxis]
    density = df[:, FEATURE_COLS['density']][::RATIO, np.newaxis]
    velocity = df[:, FEATURE_COLS['velocity']][::RATIO, np.newaxis]
    porosity = df[:, FEATURE_COLS['porosity']][::RATIO, np.newaxis]
    clay = df[:, FEATURE_COLS['clay']][::RATIO, np.newaxis]
    p0 = df[:, FEATURE_COLS['p0']][::RATIO, np.newaxis]
    pw = df[:, FEATURE_COLS['pw']][::RATIO, np.newaxis]
    
    # Calculate effective pressure (pe)
    pe = p0 - pw
    
    # Combine features
    x = np.column_stack((velocity, density, porosity, clay))
    y = pe
    
    # Normalize features
    scaler_x = MinMaxScaler()
    x = scaler_x.fit_transform(x)
    
    # Split into training and testing sets
    x_train = x[TRAIN_START:TRAIN_END, :]
    y_train = y[TRAIN_START:TRAIN_END, :]
    x_test = x[:, :]
    y_test = y[:, :]
    depth_train = depth[TRAIN_START:TRAIN_END, :]
    depth_test = depth[:, :]
    
    # Apply sparsity factor
    x_train = x_train[::FACTOR, :]
    y_train = y_train[::FACTOR, :]
    x_test = x_test[::FACTOR, :]
    y_test = y_test[::FACTOR, :]
    depth_train = depth_train[::FACTOR, :]
    depth_test = depth_test[::FACTOR, :]
    
    # Reshape for Bi-GRU (batch_size=1, channels, sequence_length)
    x_train = np.transpose(np.expand_dims(x_train, axis=0), (0, 2, 1))
    y_train = np.transpose(np.expand_dims(y_train, axis=0), (0, 2, 1))
    x_test = np.transpose(np.expand_dims(x_test, axis=0), (0, 2, 1))
    y_test = np.transpose(np.expand_dims(y_test, axis=0), (0, 2, 1))
    
    return x_train, y_train, x_test, y_test, depth_train, depth_test, scaler_x

def create_dataloaders(x_train, y_train, x_test, y_test):
    """
    Create PyTorch DataLoader objects for training and testing.
    """
    device = TRAIN_CONFIG['device']
    
    # Convert to tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Create datasets
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)
    
    return train_loader, test_loader


