"""
Configuration parameters for the pretraining model.
"""
import torch

# Data configuration
DATA_PATH = 'example.csv'
RATIO = 10  # Downsampling ratio
FACTOR = 1  # Additional sparsity factor
TRAIN_START = 0
TRAIN_END = 5000

# Feature columns (adjust according to your CSV structure)
FEATURE_COLS = {
    'depth': 0,
    'density': 1,
    'velocity': 2,
    'porosity': 3,
    'clay': 4,
    'p0': 5,
    'pw': 6
}

# Model configuration
MODEL_CONFIG = {
    'in_channels': 4,  # velocity, density, porosity, clay
    'nonlinearity': "relu",
    'cnn1_out': 8,
    'cnn2_out': 8,
    'cnn3_out': 8,
    'cnn_out_channels': 16,
    'gru_hidden_size': 8,
    'gru_num_layers': 3,
    'gru_out_hidden': 8,
    'gru_out_layers': 1,
    'output_features': 1
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 40,
    'num_epochs': 5000,
    'learning_rate': 1e-2,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'save_path': "./invert_checkpoints/pre_train.pth"
}

# Plot configuration
PLOT_CONFIG = {
    'font_family': 'Times New Roman',
    'save_dpi': 300,
    'save_paths': {
        'loss_curve': 'image/train_loss.png',
        'prediction': 'image/predicted_results.png'
    }
}