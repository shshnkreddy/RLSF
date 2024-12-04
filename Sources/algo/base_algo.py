from abc import ABC, abstractmethod
import os
import numpy as np
import torch

class Algorithm(ABC):
    def __init__(self,device, seed, gamma):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.learning_steps = 0
        self.state_shape = 0
        self.action_shape = 0
        self.device = device
        self.gamma = gamma
        self.seed = seed

    def explore(self,state):
        state = torch.from_numpy(np.array(state)).float().to(self.device)
        with torch.no_grad():
            (action,log_pi) = self.actor.sample(state)
        return action.cpu().numpy(),log_pi.cpu().numpy()

    def exploit(self,state):
        state = torch.from_numpy(np.array(state)).float().to(self.device)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()

    @abstractmethod
    def is_update(self,step):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save_models(self,save_dir):
        pass
