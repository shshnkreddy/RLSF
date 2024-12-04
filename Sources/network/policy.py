import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from .cnn import CNNHead

from .utils import build_mlp, reparameterize, evaluate_lop_pi
    

class StateIndependentPolicy(nn.Module):
    
    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh(),add_dim=0):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0]+add_dim,
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))
        self.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
            
    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)
    

class CNNStateIndependentPolicy(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh(),add_dim=0):
        
        super().__init__()
    
        self.conv = CNNHead(state_shape)

        with torch.no_grad():
            sample_in = torch.zeros(1, *state_shape)
            feature_dim = self.conv(sample_in).shape[1]

        self.linear = build_mlp(
            input_dim=feature_dim+add_dim,
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

        self.apply(self.init_weights)

        self.net = nn.Sequential(self.conv, self.linear)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
        
    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)


        
    
