"""
Training loop and model saving.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from config import TRAIN_CONFIG

def train_model(model, train_loader, test_loader):
    """
    Train the model and return training losses.
    """
    device = TRAIN_CONFIG['device']
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'])
    
    # Training loop
    train_losses = []
    num_epochs = TRAIN_CONFIG['num_epochs']
    
    print(f"Training on device: {device}")
    print(f"Number of epochs: {num_epochs}")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.5f}")
    
    # Save the trained model
    torch.save(model, TRAIN_CONFIG['save_path'])
    print(f"Model saved to {TRAIN_CONFIG['save_path']}")
    
    return train_losses