"""
Model evaluation and metric calculation.
"""
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from config import TRAIN_CONFIG

def evaluate_model(model, test_loader):
    """
    Evaluate the model on test data and return predictions.
    """
    device = TRAIN_CONFIG['device']
    model.eval()
    
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            y_pred.append(outputs)
            y_true.append(labels)
    
    # Concatenate all batches
    y_pred = torch.cat(y_pred).cpu().numpy().squeeze()
    y_true = torch.cat(y_true).cpu().numpy().squeeze()
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Print metrics
    print('\n' + '='*50)
    print('MODEL EVALUATION METRICS')
    print('='*50)
    print(f'MAE (Mean Absolute Error): {mae:.4f}')
    print(f'MSE (Mean Squared Error): {mse:.4f}')
    print(f'RMSE (Root Mean Squared Error): {rmse:.4f}')
    print(f'R² Score: {r2:.4f}')
    print(f'True values shape: {y_true.shape}')
    print(f'Predicted values shape: {y_pred.shape}')
    print('='*50)
    
    return y_true, y_pred, {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}