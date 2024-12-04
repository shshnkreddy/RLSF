import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .utils import build_mlp

class AutoEncoder(nn.Module):
    def __init__(self, state_shape, action_shape, output_size, hidden_units=(10), latent_size = 8,
     hidden_activation=nn.ReLU(), use_actions=True):
        super().__init__()
        self.use_actions = use_actions
        if(use_actions):
            input_size = state_shape[0] + action_shape[0]
        else:
            input_size = state_shape[0]
        self.encoder = build_mlp(
            input_dim=input_size,
            output_dim=latent_size,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.decoder = build_mlp(
            input_dim=latent_size,
            output_dim=output_size,
            hidden_units=hidden_units[::-1],
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

        output = self.decoder(F.relu(self.encoder(input)))
        # output = self.AvgL1Norm(output)
        
        return output
    
    def AvgL1Norm(self, x, eps=1e-8):
        return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)

    def encode(self, states, actions):
        if(self.use_actions==False):
            # actions = torch.rand(actions.shape, device=actions.device)
            input = states
        else:
            input = torch.cat([states, actions], dim=-1)

        return self.encoder(input)

