import torch 
import torch.nn as nn
from .utils import build_mlp
import torch.nn.init as init   

class MLEClassifier_network(nn.Module):
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

    def forward(self, states, actions):
        
        if(self.use_actions==False):
            # actions = torch.rand(actions.shape, device=actions.device)
            input = states
        else:
            input = torch.cat([states, actions], dim=-1)
        
        return self.net(input)
    
    def get_traj_logits(self, states, actions, max_episode_length):
        logits = self.forward(states, actions)

        # reshape to (batch_size, max_episode_length)
        logits = logits.view(-1, max_episode_length)
        logits = torch.prod(logits, dim=1).view(-1, 1)

        return logits
    
    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)