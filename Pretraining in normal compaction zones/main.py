"""
Created on Sun Feb  8 02:53:54 2026
@author: Yingtian Liu
"""
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
plt.close('all')
plt.rcParams['font.family'] = 'Times New Roman'


"""Pre-training using W1 as training set"""
df = pd.read_csv('example.csv')
df = df.values

ratio = 10
depth = df[:,0][::ratio, np.newaxis]
density = df[:,1][::ratio, np.newaxis]
velocity = df[:,2][::ratio, np.newaxis]
porosity = df[:,3][::ratio, np.newaxis]
clay = df[:,4][::ratio, np.newaxis]
p0 = df[:,5][::ratio, np.newaxis]
pw = df[:,6][::ratio, np.newaxis]
pe = p0 - pw

# Combine these variables into a matrix
x = np.column_stack((velocity, density, porosity, clay))
y = pe
scaler_x = MinMaxScaler()
x = scaler_x.fit_transform(x)
start_number = 0
end_number = 5000

  
x_train = x[start_number:end_number,:]
y_train = y[start_number:end_number,:]
x_test = x[:,:]
y_test = y[:,:]
depth_top = depth[start_number:end_number,:]
depth_bottom = depth[end_number:,:]

"""Sparsity"""
factor = 1
x_train = x_train[::factor,:]
y_train = y_train[::factor,:]
x_test = x_test[::factor,:]
y_test = y_test[::factor,:]
depth_top = depth_top[::factor,:]
depth_bottom = depth_bottom[::factor,:]

"""Bi-GRU"""
x_train = np.transpose(np.expand_dims(x_train, axis=0), (0, 2, 1))
y_train = np.transpose(np.expand_dims(y_train, axis=0), (0, 2, 1))
x_test = np.transpose(np.expand_dims(x_test, axis=0), (0, 2, 1))
y_test = np.transpose(np.expand_dims(y_test, axis=0), (0, 2, 1))

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""Training dataset"""
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

"""Test dataset"""
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
batch_size = 40
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#%% Define GRU-CNN model
class inverse_model(nn.Module):
    def __init__(self, in_channels, nonlinearity="tanh"):
        super(inverse_model, self).__init__()
        self.in_channels = in_channels
        self.activation =  nn.ReLU() if nonlinearity=="relu" else nn.Tanh()
        """ Local pattern analysis"""
        self.cnn1 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels,
                                            out_channels=8,
                                            kernel_size=5,
                                            padding="same",
                                            dilation=1),
                                   nn.GroupNorm(num_groups=4,
                                                num_channels=8))
        self.cnn2 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels,
                                           out_channels=8,
                                           kernel_size=5,
                                            padding="same",
                                           dilation=3),
                                  nn.GroupNorm(num_groups=4,
                                               num_channels=8))

        self.cnn3 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels,
                                           out_channels=8,
                                           kernel_size=5,
                                            padding="same",
                                           dilation=6),
                                  nn.GroupNorm(num_groups=4,
                                               num_channels=8))

        self.cnn = nn.Sequential(self.activation,
                                 nn.Conv1d(in_channels=24,
                                           out_channels=16,
                                           kernel_size=3,
                                            padding="same"),
                                 nn.GroupNorm(num_groups=4,
                                              num_channels=16),
                                 self.activation,

                                 nn.Conv1d(in_channels=16,
                                           out_channels=16,
                                           kernel_size=3,
                                            padding="same"),
                                 nn.GroupNorm(num_groups=4,
                                              num_channels=16),
                                 self.activation,

                                 nn.Conv1d(in_channels=16,
                                           out_channels=16,
                                           kernel_size=1),
                                 nn.GroupNorm(num_groups=4,
                                              num_channels=16),
                                 self.activation)
        """Sequence modeling"""
        self.gru = nn.GRU(input_size=self.in_channels,
                          hidden_size=8,
                          num_layers=3,
                          batch_first=True,
                          bidirectional=True)
        """Regression"""
        self.gru_out = nn.GRU(input_size=16,
                              hidden_size=8,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)
        self.out = nn.Linear(in_features=16, out_features=1)
   
    def forward(self, x):
        cnn_out1 = self.cnn1(x)
        cnn_out2 = self.cnn2(x)
        cnn_out3 = self.cnn3(x)
        cnn_out = self.cnn(torch.cat((cnn_out1,cnn_out2,cnn_out3),dim=1))
        tmp_x = x.transpose(-1, -2) 
        """gru"""
        rnn_out, _ = self.gru(tmp_x)
        rnn_out = rnn_out.transpose(-1, -2)
        x1 = rnn_out + cnn_out
        x1 = x1.transpose(-1, -2)
        x1, _ = self.gru_out(x1)
        """ Linear regression layer """
        x1 = self.out(x1)
        x1 = x1.transpose(-1,-2)
        return x1

#%% Instantiate model
"""CNN-GRU"""
print('Number of channels:', len(x[0,:]))
model = inverse_model(in_channels=len(x[0,:]), nonlinearity="relu").to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-2)
"""Calculate network parameters"""
print('Inversion network parameter count:', sum([param.nelement() for param in model.parameters()])) 
train_losses = []
num_epochs = 5000
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    if (epoch+1) % 1 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.5f}")

"""Save model"""        
torch.save(model,"./invert_checkpoints/pre_train.pth")
model.eval() 
y_pred = []
y_true = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        y_pred.append(outputs)
        y_true.append(labels)
y_train = y_train.squeeze()
y_pred = torch.cat(y_pred).cpu().numpy().squeeze()
y_true = torch.cat(y_true).cpu().numpy().squeeze()

print('y_true shape:', y_true.shape)
print('y_pred shape:', y_pred.shape)
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
print(f"MAE Score: {mae:.4f}") 
print(f"MSE Score: {mse:.4f}") 

"""Plotting"""
#%% Plot loss curve
plt.figure()        
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss', color='b')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('image/train_loss.png', dpi=300, bbox_inches='tight')  
#%% Plot fitting curve
plt.figure(figsize=(4,8))
plt.plot(y_true, depth, color='k',linestyle='-',linewidth=2)
plt.plot(y_pred, depth, color='b',linestyle='-',linewidth=2)
plt.legend(['True','pred'],loc='upper right',fontsize=15) 
plt.gca().invert_yaxis()
plt.xlabel('Mpa',fontsize=15)  
plt.ylabel('Depth(m)',fontsize=15)  
plt.ylim(depth[-1],depth[0])  
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.savefig('image/predicted_results.png', dpi=300, bbox_inches='tight')  




