import numpy as np 
from ..utils import stringify
import torch 
from abc import ABC, abstractmethod
# from sklearn.kernel_approximation import RBFSampler
# from sklearn.preprocessing import PolynomialFeatures
# import threading
import pickle

class DensityModel(ABC):
    @abstractmethod
    def add(self, states, actions=None):
        pass
    
    @abstractmethod
    def get_density(self, states, actions=None):
        pass

    @abstractmethod
    def get_weight(self, states, actions=None):
        pass

class XYHeatMap(DensityModel):
    def __init__(self, x_range, y_range, x_bins, y_bins):
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.x_range = np.linspace(x_range[0], x_range[1], x_bins)
        self.y_range = np.linspace(y_range[0], y_range[1], y_bins)
        self.heatmap = np.zeros((x_bins, y_bins))
        self.count = 0

    def add(self, states, actions=None):
        x = np.digitize(states[:, 0], self.x_range)
        y = np.digitize(states[:, 1], self.y_range)
        x = np.clip(x, 0, self.x_bins-1)
        y = np.clip(y, 0, self.y_bins-1)
        for i in range(states.shape[0]):
            self.heatmap[x[i], y[i]] += 1
        self.count += states.shape[0]

    def get_density(self, states, actions=None):
        x = np.digitize(states[:, 0], self.x_range)
        y = np.digitize(states[:, 1], self.y_range)
        x = np.clip(x, 0, self.x_bins-1)
        y = np.clip(y, 0, self.y_bins-1)
        return self.heatmap[x, y]
    
    def get_weight(self, states, actions=None):
        x = np.digitize(states[:, 0], self.x_range)
        y = np.digitize(states[:, 1], self.y_range)
        x = np.clip(x, 0, self.x_bins-1)
        y = np.clip(y, 0, self.y_bins-1)

        norm_density = self.heatmap[x, y] / self.count
        idxs_0 = np.abs(norm_density) < 1e-12
        weight = np.zeros_like(norm_density)
        weight[idxs_0] = 1e2
        weight[~idxs_0] = (1e-4)/norm_density[~idxs_0]

        return weight

class SimHash(DensityModel):
    def __init__(self, k, state_shape, device, action_shape=None, use_actions=False, feature_state_dims=None):
        self.k = k
        self.hash_table = {}
        self.use_actions = use_actions
        self.device = device
        
        self.count = 0
        self.featurize = feature_state_dims is not None
        if(feature_state_dims is not None):
            self.running_state_means = np.zeros((feature_state_dims,), dtype=np.float32)
            self.running_state_vars = np.ones((feature_state_dims,), dtype=np.float32)
            self.kernel = RBFSampler(gamma=1, n_components=feature_state_dims, random_state=101)
            
        else:
            self.running_state_means = np.zeros((state_shape[0],), dtype=np.float32)
            self.running_state_vars = np.ones((state_shape[0],), dtype=np.float32)

        if(self.featurize):
            self.embedding_size = feature_state_dims
        else:
            self.embedding_size = state_shape[0]
            if(use_actions):
                self.embedding_size += action_shape[0]
        self._w = torch.randn((self.k, self.embedding_size), device=self.device)

    def add(self, states, actions=None):
        if self.use_actions:
            states = np.concatenate((states, actions))
        if self.featurize:
            states = self.kernel.fit_transform(states)
            
        if(states.shape[0] == 0):
            return     
        self.running_state_means = (self.running_state_means * self.count + np.sum(states, axis=0)) / (self.count + states.shape[0])
        self.running_state_vars = (self.running_state_vars * self.count + np.sum((states - self.running_state_means)**2, axis=0)) / (self.count + states.shape[0])
        self.count += states.shape[0] 
        #normalize values
        states = np.clip((states - self.running_state_means) / (np.sqrt(self.running_state_vars) + 1e-8), -10.0, 10.0)
        
        hash_values = self._hash(states)

        for h in hash_values:
            if(h in self.hash_table):
                self.hash_table[h] += 1
            else:
                self.hash_table[h] = 1
        

    def get_density(self, states, actions=None):
        if(self.use_actions):
            states = np.concatenate((states, actions))
        if(self.featurize):
            states = self.kernel.fit_transform(states)
        #normalize values
        states = np.clip((states - self.running_state_means) / (np.sqrt(self.running_state_vars) + 1e-8), -10.0, 10.0)
        
        hash = self._hash(states)
        densities = np.zeros(len(states))
        for i, h in enumerate(hash):
            if(h in self.hash_table):
                densities[i] = self.hash_table[h]
        return densities
    
    def get_weight(self, states, actions=None):
        if(self.use_actions):
            states = np.concatenate((states, actions))

        densities = self.get_density(states)    
        norm_density = densities / self.count
        idxs_0 = np.abs(norm_density) < 1e-12
        weight = np.zeros_like(norm_density)
        weight[idxs_0] = 1e2
        weight[~idxs_0] = (1e-6)/norm_density[~idxs_0]

        return weight

    def _hash(self, values):
        values = values.astype(np.float32)
        values = torch.tensor(values, device=self.device, requires_grad=False)
        values = torch.sign(torch.matmul(self._w, values.T)).T
        values[values<0] = 0.0
        values = values.cpu().numpy().astype(int).tolist()

        hash_values = [None]*len(values)
        stringify(values, hash_values, 0, len(values))
        
        return hash_values
    
    def save(self, path):
        self.info_dict = {
            'w': self._w.cpu().numpy(),
            'hash_table': self.hash_table,
            'running_state_means': self.running_state_means,
            'running_state_vars': self.running_state_vars,
            'count': self.count
        }
        with open(path, 'wb') as f:
            pickle.dump(self.info_dict, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.info_dict = pickle.load(f)
            self._w = torch.tensor(self.info_dict['w'], device=self.device)
            self.hash_table = self.info_dict['hash_table']
            self.running_state_means = self.info_dict['running_state_means']
            self.running_state_vars = self.info_dict['running_state_vars']
            self.count = self.info_dict['count']
    

    
    
    