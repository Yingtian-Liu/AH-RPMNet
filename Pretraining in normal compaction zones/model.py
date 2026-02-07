"""
Model architecture definition (CNN-GRU hybrid).
"""
import torch
import torch.nn as nn
from config import MODEL_CONFIG

class InverseModel(nn.Module):
    """CNN-GRU hybrid model for pore pressure prediction."""
    
    def __init__(self, in_channels=MODEL_CONFIG['in_channels'], nonlinearity=MODEL_CONFIG['nonlinearity']):
        super(InverseModel, self).__init__()
        self.in_channels = in_channels
        self.activation = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()
        
        # Local pattern analysis (parallel CNN branches with different dilations)
        self._build_cnn_layers()
        
        # Sequence modeling (Bi-GRU)
        self._build_gru_layers()
        
        # Regression head
        self._build_regression_layer()
    
    def _build_cnn_layers(self):
        """Build parallel CNN branches with different dilation rates."""
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels,
                     out_channels=MODEL_CONFIG['cnn1_out'],
                     kernel_size=5,
                     padding="same",
                     dilation=1),
            nn.GroupNorm(num_groups=4, num_channels=MODEL_CONFIG['cnn1_out'])
        )
        
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels,
                     out_channels=MODEL_CONFIG['cnn2_out'],
                     kernel_size=5,
                     padding="same",
                     dilation=3),
            nn.GroupNorm(num_groups=4, num_channels=MODEL_CONFIG['cnn2_out'])
        )
        
        self.cnn3 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels,
                     out_channels=MODEL_CONFIG['cnn3_out'],
                     kernel_size=5,
                     padding="same",
                     dilation=6),
            nn.GroupNorm(num_groups=4, num_channels=MODEL_CONFIG['cnn3_out'])
        )
        
        # Combined CNN processing
        self.cnn = nn.Sequential(
            self.activation,
            nn.Conv1d(in_channels=MODEL_CONFIG['cnn1_out'] + MODEL_CONFIG['cnn2_out'] + MODEL_CONFIG['cnn3_out'],
                     out_channels=MODEL_CONFIG['cnn_out_channels'],
                     kernel_size=3,
                     padding="same"),
            nn.GroupNorm(num_groups=4, num_channels=MODEL_CONFIG['cnn_out_channels']),
            self.activation,
            nn.Conv1d(in_channels=MODEL_CONFIG['cnn_out_channels'],
                     out_channels=MODEL_CONFIG['cnn_out_channels'],
                     kernel_size=3,
                     padding="same"),
            nn.GroupNorm(num_groups=4, num_channels=MODEL_CONFIG['cnn_out_channels']),
            self.activation,
            nn.Conv1d(in_channels=MODEL_CONFIG['cnn_out_channels'],
                     out_channels=MODEL_CONFIG['cnn_out_channels'],
                     kernel_size=1),
            nn.GroupNorm(num_groups=4, num_channels=MODEL_CONFIG['cnn_out_channels']),
            self.activation
        )
    
    def _build_gru_layers(self):
        """Build GRU layers for sequence modeling."""
        self.gru = nn.GRU(
            input_size=self.in_channels,
            hidden_size=MODEL_CONFIG['gru_hidden_size'],
            num_layers=MODEL_CONFIG['gru_num_layers'],
            batch_first=True,
            bidirectional=True
        )
        
        self.gru_out = nn.GRU(
            input_size=MODEL_CONFIG['cnn_out_channels'],
            hidden_size=MODEL_CONFIG['gru_out_hidden'],
            num_layers=MODEL_CONFIG['gru_out_layers'],
            batch_first=True,
            bidirectional=True
        )
    
    def _build_regression_layer(self):
        """Build final regression layer."""
        self.out = nn.Linear(
            in_features=MODEL_CONFIG['gru_out_hidden'] * 2,  # Bidirectional
            out_features=MODEL_CONFIG['output_features']
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        # CNN branches
        cnn_out1 = self.cnn1(x)
        cnn_out2 = self.cnn2(x)
        cnn_out3 = self.cnn3(x)
        
        # Concatenate and process through combined CNN
        cnn_out = self.cnn(torch.cat((cnn_out1, cnn_out2, cnn_out3), dim=1))
        
        # GRU processing
        tmp_x = x.transpose(-1, -2)
        rnn_out, _ = self.gru(tmp_x)
        rnn_out = rnn_out.transpose(-1, -2)
        
        # Combine CNN and GRU outputs
        x1 = rnn_out + cnn_out
        x1 = x1.transpose(-1, -2)
        
        # Final GRU and regression
        x1, _ = self.gru_out(x1)
        x1 = self.out(x1)
        x1 = x1.transpose(-1, -2)
        
        return x1


def get_model(in_channels=None):
    """
    Instantiate and return the model.
    """
    if in_channels is None:
        in_channels = MODEL_CONFIG['in_channels']
    
    model = InverseModel(in_channels=in_channels, nonlinearity=MODEL_CONFIG['nonlinearity'])
    
    # Calculate and print parameter count
    param_count = sum([param.nelement() for param in model.parameters()])
    print(f'Inversion network parameter count: {param_count:,}')
    
    return model

