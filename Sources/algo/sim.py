import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from torch.optim import Adam
import wandb

from Sources.algo.ppo import PPO_lag,RolloutBuffer_PPO_lag,StateIndependentPolicy,\
    StateFunction
from Sources.network import Classifier_network

class SIM(PPO_lag):
    def __init__(self,env_name,expert_actor,exp_good_buffer,exp_bad_buffer, state_shape, action_shape, device, seed, gamma,cost_gamma,
        buffer_size, mix, hidden_units_actor, hidden_units_critic,units_clfs,batch_size,
        lr_actor, lr_critic,lr_cost_critic,lr_penalty,lr_clfs, epoch_ppo,epoch_clfs, clip_eps, lambd, coef_ent, 
        max_grad_norm,reward_factor,max_episode_length,cost_limit,risk_level,
        num_envs,dynamic_good,min_good,max_bad,
        conf_coef,tanh_conf,start_bad, primarive=True, wandb_log=False, mode='SIM'):
        super().__init__(env_name, state_shape, action_shape, device, seed, gamma,cost_gamma,
        buffer_size, mix, hidden_units_actor, hidden_units_critic,units_clfs,batch_size,
        lr_actor, lr_critic,lr_cost_critic,lr_penalty,lr_clfs, epoch_ppo,epoch_clfs, clip_eps, lambd, coef_ent, 
        max_grad_norm,reward_factor, max_episode_length, cost_limit, risk_level,
        num_envs, wandb_log, False)

        self.expert_actor = expert_actor
        self.exp_good_buffer = exp_good_buffer
        self.exp_bad_buffer = exp_bad_buffer
        self.batch_size=batch_size
        self.dynamic_good = dynamic_good
        self.min_good = min_good
        self.max_bad = max_bad
        self.conf_coef = conf_coef
        self.tanh_conf = tanh_conf
        self.start_bad = start_bad
        self.new_good = 0
        self.new_bad = 0
        self.mode = mode

        if (primarive):
            self.clfs = Classifier_network(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_clfs,
            hidden_activation=nn.ReLU()
            ).to(device)

            self.optim_clfs = Adam(self.clfs.parameters(), lr=lr_clfs)

        self.expert_return_cost = []
        self.expert_return_reward = []
        self.expert_sucess_rate = []

    def get_expert_return_threshold(self):
        return self.min_good

    def get_return_threshold(self):
        if (len(self.return_reward)==0):
            return self.min_good
        if (self.dynamic_good):
            return min(
                np.mean(self.return_reward)+2*max(1.0,np.std(self.return_reward)),
                       self.min_good)
        else:
            return self.min_good

    def get_bad_threshold(self):
        if (len(self.return_reward)==0):
            return 0.0
        return min(
                np.mean(self.return_reward)-max(1.0,np.std(self.return_reward)),
                   self.max_bad)
    
    def expert_exploit(self,state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.expert_actor(state)
        return action.cpu().numpy()
    
    def expert_step(self, env, state, ep_len):
        action = self.expert_exploit(state)
        next_state, reward, c, done, _, _  = env.step(action)
        for idx in range(self.num_envs):
            ep_len[idx] += 1
            mask = False if ep_len[idx] >= self.max_episode_length else done[idx]
            self.tmp_buffer[idx].append((state[idx], action[idx], reward[idx] * self.reward_factor,
            c[idx], mask, None, next_state[idx]))
            self.tmp_return_cost[idx] += c[idx]
            self.tmp_return_reward[idx] += reward[idx]
            if (self.max_episode_length and ep_len[idx]>=self.max_episode_length):
                done[idx] = True

        for idx in range(self.num_envs):
            if (done[idx] == False):
                continue
            is_good = False
            if (self.tmp_return_reward[idx]>self.get_expert_return_threshold() 
                    and self.tmp_return_cost[idx]<self.cost_limit):
                is_good = True
            if (is_good):
                for (tmp_state,tmp_action,tmp_reward,tmp_c,tmp_mask,tmp_log_pi,tmp_next_state) in self.tmp_buffer[idx]:
                        self.exp_good_buffer.append_roll(tmp_state,tmp_action,tmp_next_state,
                                np.array([tmp_reward]),np.array([tmp_c]),np.array([float(tmp_mask)]))
                    
            self.tmp_buffer[idx] = []
            if (is_good):
                self.expert_sucess_rate.append(1.0)
                self.expert_return_cost.append(self.tmp_return_cost[idx])
                self.expert_return_reward.append(self.tmp_return_reward[idx])
            else:
                self.expert_sucess_rate.append(0.0)
            self.tmp_return_cost[idx] = 0
            self.tmp_return_reward[idx] = 0

        if(done[0]):
            next_state,_ = env.reset()
            ep_len = [0 for _ in range(self.num_envs)]
            
        return next_state, ep_len,np.mean(self.expert_return_reward),\
        np.mean(self.expert_return_cost),np.mean(self.expert_sucess_rate),len(self.expert_sucess_rate)
    
    def step(self, env, state, ep_len):
        action, log_pi = self.explore(state)
        next_state, reward, c, done, _, _  = env.step(action)
        for idx in range(self.num_envs):
            ep_len[idx] += 1
            mask = False if ep_len[idx] >= self.max_episode_length else done[idx]
            self.tmp_buffer[idx].append((state[idx], action[idx], reward[idx] * self.reward_factor,
                                         c[idx], mask, log_pi[idx], next_state[idx]))
            self.tmp_return_cost[idx] += c[idx]
            self.tmp_return_reward[idx] += reward[idx]
            if (self.max_episode_length and ep_len[idx]>=self.max_episode_length):
                done[idx] = True

            if done[idx]:
                is_good = False
                is_bad = False
                if('SIM' in self.mode):
                    if (self.tmp_return_reward[idx]>self.get_return_threshold() 
                            and self.tmp_return_cost[idx]<=self.cost_limit):
                        self.new_good += 1
                        is_good = True
                    if (self.tmp_return_reward[idx]<self.get_bad_threshold() or self.tmp_return_cost[idx]>self.cost_limit):
                        self.new_bad += 1
                        is_bad = True
                elif(self.mode=='DGB'):
                    if (self.tmp_return_cost[idx]<=self.cost_limit):
                        self.new_good += 1
                        is_good = True
                    if (self.tmp_return_cost[idx]>self.cost_limit):
                        self.new_bad += 1
                        is_bad = True

                for (tmp_state,tmp_action,tmp_reward,tmp_c,tmp_mask,tmp_log_pi,tmp_next_state) in self.tmp_buffer[idx]:
                    self.buffer.append(tmp_state, tmp_action, tmp_reward,self.tmp_return_reward[idx], tmp_c, tmp_mask, tmp_log_pi, tmp_next_state)
                    if (is_good):
                        self.exp_good_buffer.append_roll(tmp_state,tmp_action,tmp_next_state,
                                        np.array([tmp_reward]),np.array([tmp_c]),np.array([float(tmp_mask)]))
                    elif (is_bad):
                        self.exp_bad_buffer.append_roll(tmp_state,tmp_action,tmp_next_state,
                                        np.array([tmp_reward]),np.array([tmp_c]),np.array([float(tmp_mask)]))
                        
                self.tmp_buffer[idx] = []
                self.return_cost.append(self.tmp_return_cost[idx])
                self.return_reward.append(self.tmp_return_reward[idx])
                self.tmp_return_cost[idx] = 0
                self.tmp_return_reward[idx] = 0

        if(done[0]):
            next_state,_ = env.reset()
            ep_len = [0 for _ in range(self.num_envs)]

        return next_state, ep_len
    
    def update(self):
        for _ in range(self.epoch_clfs):
            
            bad_states, bad_actions,_,_,_,_ = self.exp_bad_buffer.sample_roll(self.batch_size)
            pi_states, pi_actions, _,pi_returns, _, _, _, _ = self.buffer.sample(self.batch_size)
            good_states, good_actions,_,_,_,_ = self.exp_good_buffer.sample_roll(self.batch_size)

            if(bad_states is None or good_states is None or pi_states is None):
                continue
            self.update_clfs(   bad_states,bad_actions,
                                good_states,good_actions,
                                pi_states, pi_actions,pi_returns)

        states, actions, env_rewards, total_env_rewards, costs, dones, log_pis, next_states = self.buffer.get()
        env_rewards = env_rewards.clamp(min=-3.0,max=3.0)
        if (self.exp_bad_buffer.roll_n>=self.start_bad*self.max_episode_length):
            confidents = self.clfs.get_confident_sigmoid(states,actions).detach()
        else:
            confidents = torch.tensor(0.0)

        if(self.mode=='SIM'):
            env_rewards *= 0
            costs *= 0
        elif(self.mode=='DGB'):
            costs *= 0

        conf_reward = self.conf_coef*confidents
        rewards = env_rewards + conf_reward

        print(f'[Train] R: {np.mean(self.return_reward):.2f}, C: {np.mean(self.return_cost):.2f}, '+
              f'newG: {self.new_good}, newB: {self.new_bad}, mConf: {confidents.mean().item():.5f}, '+
              f'RConf: {conf_reward.mean().item():.5f}, Renv: {env_rewards.mean().item():.5f}')
        
        if(self.wandb_log):
            wandb.log({
                'train/R': np.mean(self.return_reward),
                'train/C': np.mean(self.return_cost),
                'train/mConf': confidents.mean().item(),
                'train/RConf': conf_reward.mean().item(),
                'train/Renv': env_rewards.mean().item(), 
                'buffer/newG': self.new_good,
                'buffer/newB': self.new_bad,
            })
        
        self.update_ppo(
            states, actions, rewards, costs, dones, log_pis, next_states)
        self.return_cost = []
        self.return_reward = []
        self.new_good = 0
        self.new_bad = 0
        
    def update_clfs(self, 
                    bad_states,bad_actions,
                    good_states,good_actions,
                    pi_states, pi_actions,pi_returns):
        bad_logits = self.clfs(bad_states,bad_actions)
        bad_loss = (1/8*bad_logits**2 + bad_logits).mean()
        good_logits = self.clfs(good_states,good_actions)
        good_loss = (1/8*good_logits**2 - good_logits).mean()
        pi_logits = self.clfs(pi_states,pi_actions)[pi_returns>=np.mean(self.return_reward)+np.std(self.return_reward)]
        pi_loss = (1/8*pi_logits**2 - pi_logits).mean()
        self.optim_clfs.zero_grad()
        (bad_loss+good_loss+pi_loss).backward()
        nn.utils.clip_grad_norm_(self.clfs.parameters(), self.max_grad_norm)
        self.optim_clfs.step()