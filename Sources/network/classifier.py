import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from .utils import build_mlp

class Classifier_network(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
     hidden_activation=nn.Tanh(), use_actions=True):
        super().__init__()
        self.use_actions = use_actions
        if(use_actions):
            input_size = state_shape[0] + action_shape[0]
        else:
            input_size = state_shape[0]
        self.net = build_mlp(
            input_dim=input_size,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
    
    def forward(self, states, actions):
        if(self.use_actions==False):
            # actions = torch.rand(actions.shape, device=actions.device)
            input = states
        else:
            input = torch.cat([states, actions], dim=-1)
        
        return self.net(input)

    def get_confident_sigmoid(self, states, actions):
        input = torch.cat([states, actions], dim=-1)
        return F.sigmoid(self.net(input))

    def get_confident_tanh(self, states, actions):
        input = torch.cat([states, actions], dim=-1)
        return F.tanh(self.net(input))
    
    def get_optimal_ratio(self, states, actions):
        logits = self.get_confident_sigmoid(states, actions)
        logits = logits.clamp(min=1e-3, max=1 - 1e-3) #numerical stability
        return torch.log(logits / (1 - logits))
    
    

         