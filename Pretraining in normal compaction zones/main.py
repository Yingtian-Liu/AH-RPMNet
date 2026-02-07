"""
Main script for pretraining the pore pressure prediction model.
"""
from dataloader import load_and_preprocess_data, create_dataloaders
from model import get_model
from trainer import train_model
from evaluator import evaluate_model
from plot_utils import setup_plotting, plot_loss_curve, plot_predictions, plot_feature_distributions
from config import TRAIN_CONFIG, MODEL_CONFIG

def main():
    """
    Main training pipeline for the pretraining model.
    """
    # Setup
    setup_plotting()
    print("="*60)
    print("PORE PRESSURE PREDICTION - PRETRAINING PHASE")
    print("="*60)
    
    # 1. Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    x_train, y_train, x_test, y_test, depth_train, depth_test, scaler = load_and_preprocess_data()
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Testing data shape: {x_test.shape}")
    print(f"Depth data shape: {depth_test.shape}")
    
    # 2. Create data loaders
    print("\n2. Creating data loaders...")
    train_loader, test_loader = create_dataloaders(x_train, y_train, x_test, y_test)
    
    # 3. Initialize model
    print("\n3. Initializing model...")
    model = get_model(in_channels=MODEL_CONFIG['in_channels'])
    
    # 4. Train model
    print("\n4. Training model...")
    train_losses = train_model(model, train_loader, test_loader)
    
    # 5. Evaluate model
    print("\n5. Evaluating model...")
    y_true, y_pred, metrics = evaluate_model(model, test_loader)
    
    # 6. Plot results
    print("\n6. Generating plots...")
    
    # Plot training loss
    plot_loss_curve(train_losses)
    
    # Plot predictions vs true values
    # Flatten depth for plotting
    depth_flat = depth_test.squeeze()
    plot_predictions(y_true, y_pred, depth_flat)
    
    # Plot feature distributions (optional)
    try:
        plot_feature_distributions(x_train)
    except Exception as e:
        print(f"Note: Could not plot feature distributions: {e}")
    
    print("\n" + "="*60)
    print("PRETRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved to: {TRAIN_CONFIG['save_path']}")
    print(f"Key metric - RMSE: {metrics['RMSE']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()