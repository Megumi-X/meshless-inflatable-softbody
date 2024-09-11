import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network_size = 1024

class DeepSDFWithCode(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(DeepSDFWithCode, self).__init__()
        self.network = nn.Sequential(
            weight_norm(nn.Linear(3, network_size)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(network_size, network_size)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(network_size, network_size)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(network_size, network_size)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(network_size, network_size)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(network_size, network_size)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(network_size, network_size)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(network_size, network_size)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(network_size, 1)),
        )
    
    def forward(self, coords):
        return self.network(coords)