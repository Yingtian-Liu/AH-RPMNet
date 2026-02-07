"""
Plotting and visualization functions.
"""
import matplotlib.pyplot as plt
import numpy as np
from config import PLOT_CONFIG

def setup_plotting():
    """Initialize matplotlib settings."""
    plt.rcParams['font.family'] = PLOT_CONFIG['font_family']
    plt.close('all')

def plot_loss_curve(train_losses, save_path=None):
    """
    Plot training loss curve.
    """
    if save_path is None:
        save_path = PLOT_CONFIG['save_paths']['loss_curve']
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 
             label='Training Loss', color='b', linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=PLOT_CONFIG['save_dpi'], bbox_inches='tight')
    plt.show()
    
    print(f"Loss curve saved to {save_path}")

def plot_predictions(y_true, y_pred, depth, save_path=None):
    """
    Plot true vs predicted values against depth.
    """
    if save_path is None:
        save_path = PLOT_CONFIG['save_paths']['prediction']
    
    plt.figure(figsize=(6, 10))
    
    # Plot curves
    plt.plot(y_true, depth, color='k', linestyle='-', linewidth=2.5, label='True')
    plt.plot(y_pred, depth, color='b', linestyle='-', linewidth=2, label='Predicted', alpha=0.8)
    
    # Formatting
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
    plt.gca().invert_yaxis()  # Depth increases downward
    plt.xlabel('Effective Pressure (MPa)', fontsize=12)
    plt.ylabel('Depth (m)', fontsize=12)
    plt.ylim(depth[-1], depth[0])
    plt.tick_params(labelsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.title('True vs Predicted Pressure', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=PLOT_CONFIG['save_dpi'], bbox_inches='tight')
    plt.show()
    
    print(f"Prediction plot saved to {save_path}")

def plot_feature_distributions(x_train, feature_names=None):
    """
    Plot distributions of input features.
    """
    if feature_names is None:
        feature_names = ['Velocity', 'Density', 'Porosity', 'Clay Content']
    
    x_train_flat = x_train.squeeze().T  # Shape: (features, samples)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(min(4, x_train_flat.shape[0])):
        axes[i].hist(x_train_flat[i], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[i].set_xlabel(feature_names[i], fontsize=11)
        axes[i].set_ylabel('Frequency', fontsize=11)
        axes[i].set_title(f'{feature_names[i]} Distribution', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('image/feature_distributions.png', dpi=PLOT_CONFIG['save_dpi'], bbox_inches='tight')
    plt.show()
    
    