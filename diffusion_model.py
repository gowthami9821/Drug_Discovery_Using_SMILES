# diffusion_model.py
import torch
import torch.nn as nn

class SimpleDiffusionModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024):
        super(SimpleDiffusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        return self.net(x)
