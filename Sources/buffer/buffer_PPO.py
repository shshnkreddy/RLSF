import os
import numpy as np
import torch

class Trajectory_Buffer_Continuous:

    def __init__(self, buffer_size, state_shape, action_shape, device, aug_state_shape=None, priority=False, tau=0.1):
        if (buffer_size):
            self.roll_n = 0
            self.roll_p = 0
            self.buffer_size = buffer_size
            self.use_aug = aug_state_shape is not None
            self.roll_states = torch.empty(
                (buffer_size,*state_shape), dtype=torch.float, device=device)
            self.roll_next_states = torch.empty(
                (buffer_size,*state_shape), dtype=torch.float, device=device)
            self.roll_actions = torch.empty(
                (buffer_size,*action_shape), dtype=torch.float, device=device)
            self.roll_rewards = torch.empty(
                (buffer_size,1), dtype=torch.float, device=device)
            self.roll_costs = torch.empty(
                (buffer_size,1), dtype=torch.float, device=device)
            self.roll_dones = torch.empty(
                (buffer_size,1), dtype=torch.float, device=device)
            if(aug_state_shape is None):
                aug_state_shape = state_shape
            self.roll_aug_states = torch.empty(
                (buffer_size,*aug_state_shape), dtype=torch.float, device=device)
            self.priority = priority
            self.tau = tau            
    
    def sample_roll(self, batch_size):
        if (self.roll_n>0 and self.buffer_size>0):
            idxes = np.random.randint(low=0, high=self.roll_n, size=batch_size)
    
            if(self.use_aug):
                return (
                    self.roll_states[idxes].to('cuda'),
                    self.roll_actions[idxes].to('cuda'),
                    self.roll_next_states[idxes].to('cuda'),
                    self.roll_rewards[idxes].to('cuda'),
                    self.roll_costs[idxes].to('cuda'),
                    self.roll_dones[idxes].to('cuda'),
                    self.roll_aug_states[idxes].to('cuda'),
                )
            return (
                self.roll_states[idxes].to('cuda'),
                self.roll_actions[idxes].to('cuda'),
                self.roll_next_states[idxes].to('cuda'),
                self.roll_rewards[idxes].to('cuda'),
                self.roll_costs[idxes].to('cuda'),
                self.roll_dones[idxes].to('cuda'),
            )
        else:
            if(self.use_aug):
                return(
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            return(
                None,
                None,
                None,
                None,
                None,
                None,
            )
        
    def sample_cont(self, batch_size):
        #pick random contiguous samples from the buffer
        start = np.random.randint(0, self.roll_n - batch_size)
        idxes = slice(start, start + batch_size)

        if(self.use_aug):
            return (
                self.roll_states[idxes].to('cuda'),
                self.roll_actions[idxes].to('cuda'),
                self.roll_next_states[idxes].to('cuda'),
                self.roll_rewards[idxes].to('cuda'),
                self.roll_costs[idxes].to('cuda'),
                self.roll_dones[idxes].to('cuda'),
                self.roll_aug_states[idxes].to('cuda'),
            )
        return (
            self.roll_states[idxes].to('cuda'),
            self.roll_actions[idxes].to('cuda'),
            self.roll_next_states[idxes].to('cuda'),
            self.roll_rewards[idxes].to('cuda'),
            self.roll_costs[idxes].to('cuda'),
            self.roll_dones[idxes].to('cuda'),
        )
    
    def append_roll(self,state,action,next_state,reward,cost,done,aug_state=None):
        assert self.buffer_size>0
        self.roll_states[self.roll_p].copy_(torch.from_numpy(state))
        self.roll_actions[self.roll_p].copy_(torch.from_numpy(action))
        self.roll_next_states[self.roll_p].copy_(torch.from_numpy(next_state))
        self.roll_rewards[self.roll_p].copy_(torch.from_numpy(reward))
        self.roll_costs[self.roll_p].copy_(torch.from_numpy(cost))
        self.roll_dones[self.roll_p].copy_(torch.from_numpy(done))
        if(self.use_aug):
            self.roll_aug_states[self.roll_p].copy_(torch.from_numpy(aug_state))

        self.roll_p = (self.roll_p + 1) % self.buffer_size
        self.roll_n = min(self.roll_n + 1, self.buffer_size)

class RolloutBuffer_PPO_lag:
    
    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1, aug_state_shape=None):
        self._n = 0
        self._p = 0
        self._r = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.state_shape = state_shape
        self.action_shape = action_shape
        self.use_aug = aug_state_shape is not None
        self.aug_state_shape = aug_state_shape
        if(aug_state_shape is None):
            self.aug_state_shape = state_shape
        self.device = device

        self.reset()
        
    def reset(self):
        self._n = 0
        self._p = 0

        self.states = torch.empty(
            (self.total_size, *self.state_shape), dtype=torch.float, device='cpu')
        self.actions = torch.empty(
            (self.total_size, *self.action_shape), dtype=torch.float, device='cpu')
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device='cpu')
        self.total_rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device='cpu')
        self.costs = torch.empty(
            (self.total_size, 1), dtype=torch.float, device='cpu')
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device='cpu')
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device='cpu')
        self.next_states = torch.empty(
            (self.total_size, *self.state_shape), dtype=torch.float, device='cpu')
        self.aug_states = torch.empty(  
            (self.total_size, *self.aug_state_shape), dtype=torch.float, device='cpu')


    def append(self, state, action, reward, total_reward, cost, done, log_pi, next_state, aug_state=None):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.total_rewards[self._p] = float(total_reward)
        self.costs[self._p] = float(cost)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        if(self.use_aug):
            self.aug_states[self._p].copy_(torch.from_numpy(aug_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self, cuda=True):
        start = 0
        idxes = slice(start, self._n)

        if(not cuda):
            if(self.use_aug):
                return (
                    self.states[idxes],
                    self.actions[idxes],
                    self.rewards[idxes],
                    self.total_rewards[idxes],
                    self.costs[idxes],
                    self.dones[idxes],
                    self.log_pis[idxes],
                    self.next_states[idxes],
                    self.aug_states[idxes]
                )
            return (
                self.states[idxes],
                self.actions[idxes],
                self.rewards[idxes],
                self.total_rewards[idxes],
                self.costs[idxes],
                self.dones[idxes],
                self.log_pis[idxes],
                self.next_states[idxes]
            )
        
        if(self.use_aug):
            return (
                self.states[idxes].to('cuda'),
                self.actions[idxes].to('cuda'),
                self.rewards[idxes].to('cuda'),
                self.total_rewards[idxes].to('cuda'),
                self.costs[idxes].to('cuda'),
                self.dones[idxes].to('cuda'),
                self.log_pis[idxes].to('cuda'),
                self.next_states[idxes].to('cuda'),
                self.aug_states[idxes].to('cuda')
            )
        return (
            self.states[idxes].to('cuda'),
            self.actions[idxes].to('cuda'),
            self.rewards[idxes].to('cuda'),
            self.total_rewards[idxes].to('cuda'),
            self.costs[idxes].to('cuda'),
            self.dones[idxes].to('cuda'),
            self.log_pis[idxes].to('cuda'),
            self.next_states[idxes].to('cuda')
        )

    def sample(self, batch_size):
        #pick random contiguous samples from the buffer
        start = np.random.randint(0, self._n - batch_size)
        idxes = slice(start, start + batch_size)

        if(self.use_aug):
            return (
                self.states[idxes].to('cuda'),
                self.actions[idxes].to('cuda'),
                self.rewards[idxes].to('cuda'),
                self.total_rewards[idxes].to('cuda'),
                self.costs[idxes].to('cuda'),
                self.dones[idxes].to('cuda'),
                self.log_pis[idxes].to('cuda'),
                self.next_states[idxes].to('cuda'),
                self.aug_states[idxes].to('cuda')
            )
        return (
            self.states[idxes].to('cuda'),
            self.actions[idxes].to('cuda'),
            self.rewards[idxes].to('cuda'),
            self.total_rewards[idxes].to('cuda'),
            self.costs[idxes].to('cuda'),
            self.dones[idxes].to('cuda'),
            self.log_pis[idxes].to('cuda'),
            self.next_states[idxes].to('cuda')
        )
    
