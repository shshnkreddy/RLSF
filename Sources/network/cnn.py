import torch 
import torch.nn as nn

class CNNHead(nn.Module):
    def __init__(self, state_shape):
        super(CNNHead, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(state_shape[0], 32, kernel_size=4, stride=1),
            # nn.ReLU(),
            nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.ReLU(), 
            nn.Flatten()
        )
        
    def forward(self, x):
        return self.conv(x)

    