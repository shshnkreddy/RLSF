import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from torch.optim import Adam
import wandb
from Sources.algo.ppo import PPO_lag
from Sources.network import Classifier_network
import threading
from Sources.utils import gen_aug_states, compute_train_stats, hinge_loss
from Sources.buffer import RolloutBuffer_PPO_lag

class PREFIM(PPO_lag):
    def __init__(self,env_name,exp_good_buffer, exp_bad_buffer, tmp_query_buffer, state_shape, action_shape, device, seed, gamma,cost_gamma,
        buffer_size, mix, hidden_units_actor, hidden_units_critic, units_clfs, batch_size,
        lr_actor, lr_critic, lr_cost_critic, lr_penalty, lr_clfs, epoch_ppo,epoch_clfs, clip_eps, lambd, coef_ent, 
        max_grad_norm, reward_factor, max_episode_length, env_cost_limit, risk_level,
        num_envs, start_bad, wandb_log, alpha, clip_dev, n_ensemble, segment_length, class_prob, 
        aug_state, aug_state_shape, pos_weight, 
        strat, encode_action, warm_start_steps, hash_map, over_sample, hinge_coeff, conv):
        super().__init__(env_name, state_shape, action_shape, device, seed, gamma,cost_gamma,
        buffer_size, mix, hidden_units_actor, hidden_units_critic,units_clfs,batch_size,
        lr_actor, lr_critic,lr_cost_critic,lr_penalty,lr_clfs, epoch_ppo,epoch_clfs, clip_eps, lambd, coef_ent, 
        max_grad_norm,reward_factor, max_episode_length, alpha, risk_level,
        num_envs, wandb_log, conv)

        self.env_name = env_name
        self.exp_good_buffer = exp_good_buffer
        self.exp_bad_buffer = exp_bad_buffer    

        self.batch_size = batch_size
        self.start_bad = start_bad
        self.warm_start_steps = warm_start_steps
        self.new_good = 0
        self.new_bad = 0
        self.n_queries = 0
        self.n_trajs_queried = 0

        self.alpha = alpha
        self.env_cost_limit = env_cost_limit
        self.mini_batch_size = batch_size
        self.clip_dev = clip_dev
        self.n_ensemble = n_ensemble
        self.class_prob = class_prob
        self.aug_state = aug_state
        self.aug_state_shape = aug_state_shape
        self.hash_map = hash_map

        self.strat = strat

        self.n_step = 0

        self.pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
        self.over_sample = over_sample

        self.grad_norm = 0.0

        if(self.aug_state):
            self.buffer = RolloutBuffer_PPO_lag(
                buffer_size=buffer_size,
                state_shape=state_shape,
                action_shape=action_shape,
                device=device,
                mix=mix,
                aug_state_shape=aug_state_shape
            )
        self.clfs = []
        self.optim_clfs = []
        for _ in range(self.n_ensemble):
            if(self.aug_state):
                self.clfs.append(Classifier_network(
                    state_shape=aug_state_shape,
                    action_shape=action_shape,
                    hidden_units=units_clfs,
                    hidden_activation=nn.ReLU(), 
                    use_actions=encode_action
                ).to(device))
            else:
                self.clfs.append(Classifier_network(
                    state_shape=state_shape,
                    action_shape=action_shape,
                    hidden_units=units_clfs,
                    hidden_activation=nn.ReLU(),
                    use_actions=encode_action
                ).to(device))

            self.optim_clfs.append(Adam(self.clfs[-1].parameters(), lr=lr_clfs))

        
        self.optim_penalty = Adam([self.penalty], lr=lr_penalty)
        self.hinge_coeff = hinge_coeff

        if(segment_length is None):
            self.segment_length = max_episode_length

        else:
            self.segment_length = segment_length
            assert max_episode_length%segment_length==0, 'max_episode_length should be divisible by segment_length'

        self.tmp_return_cost = [[] for _ in range(self.num_envs)]
        
        self.tmp_query_buffer = tmp_query_buffer
        
    def step(self, env, state, ep_len, n_step):
        self.n_step = n_step
        action, log_pi = self.explore(state)
        next_state, reward, c, done, truncated, info  = env.step(action)
        
        for idx in range(self.num_envs):
            ep_len[idx] += 1
            mask = False if ep_len[idx] >= self.max_episode_length else done[idx]
            if(self.aug_state):
                self.tmp_buffer[idx].append((state[idx], action[idx], reward[idx] * self.reward_factor,
                                        c[idx], mask, log_pi[idx], next_state[idx], gen_aug_states(state[idx], self.env_name)))
            else:
                self.tmp_buffer[idx].append((state[idx], action[idx], reward[idx] * self.reward_factor,
                                        c[idx], mask, log_pi[idx], next_state[idx]))
            self.tmp_return_cost[idx].append(c[idx])
            self.tmp_return_reward[idx] += reward[idx]
            if ((self.max_episode_length and ep_len[idx]>=self.max_episode_length) or truncated[idx]):
                done[idx] = True

            if done[idx]:
                
                warmup = True if self.n_step < self.warm_start_steps else False

                if(not warmup):
                                        
                    traj_states = []
                    traj_actions = []
                    traj_rewards = []
                    traj_costs = []

                    if(self.aug_state):
                        for tmp_state,tmp_action,tmp_reward,tmp_c,_,_,_, tmp_aug_state in self.tmp_buffer[idx]:
                            traj_states.append(tmp_aug_state) 
                            traj_actions.append(tmp_action)
                            traj_rewards.append(tmp_reward)
                            traj_costs.append(tmp_c)

                    else:
                        for tmp_state,tmp_action,tmp_reward,tmp_c,_,_,_ in self.tmp_buffer[idx]:
                            traj_states.append(tmp_state)
                            traj_actions.append(tmp_action)
                            traj_rewards.append(tmp_reward)
                            traj_costs.append(tmp_c)

                    if(self.strat=='novel'):
                        # Check if trajectory is novel
                        is_novel = self.get_novelty(traj_states, traj_actions) or self.exp_bad_buffer.roll_n<self.start_bad
                        self.tmp_query_buffer.add(traj_states, traj_actions, traj_rewards, traj_costs, novel=is_novel)
                    
                    elif(self.strat=='entropy'):  
                        _states = torch.tensor(np.array(traj_states), dtype=torch.float32, device=self.device)  
                        _actions = torch.tensor(np.array(traj_actions), dtype=torch.float32, device=self.device)

                        cprobs = torch.sigmoid(self.clfs[0](_states,_actions)).detach().cpu().numpy()
                        entropy = (-cprobs*np.log(cprobs+1e-6) - (1-cprobs)*np.log(1-cprobs+1e-6)).mean()

                        self.tmp_query_buffer.add(traj_states, traj_actions, traj_rewards, traj_costs, entropy=entropy)

                    else:
                        self.tmp_query_buffer.add(traj_states, traj_actions, traj_rewards, traj_costs)

                if(self.aug_state):
                    for tmp_state,tmp_action,tmp_reward,tmp_c,tmp_mask,tmp_log_pi,tmp_next_state, tmp_aug_state in self.tmp_buffer[idx]:
                        self.buffer.append(tmp_state, tmp_action, tmp_reward, self.tmp_return_reward[idx], tmp_c, tmp_mask, tmp_log_pi, tmp_next_state, tmp_aug_state)

                else:
                    for tmp_state,tmp_action,tmp_reward,tmp_c,tmp_mask,tmp_log_pi,tmp_next_state in self.tmp_buffer[idx]:
                        self.buffer.append(tmp_state, tmp_action, tmp_reward, self.tmp_return_reward[idx], tmp_c, tmp_mask, tmp_log_pi, tmp_next_state)
                
                self.tmp_buffer[idx] = []
                self.return_cost.append(np.sum(self.tmp_return_cost[idx]))
                self.return_reward.append(self.tmp_return_reward[idx])
                self.tmp_return_cost[idx] = []
                self.tmp_return_reward[idx] = 0
                ep_len[idx] = 0

        return next_state, ep_len
    
    def update(self):

        self.tmp_copy = self.tmp_query_buffer.trajs.copy()
        
        good_states, good_actions, bad_states, bad_actions, n_queries, n_trajs_queried, ratio = self.tmp_query_buffer.query_user(strat=self.strat)
        self.n_queries += n_queries
        self.n_trajs_queried += n_trajs_queried

        if(len(good_states) > 0 or len(bad_states) > 0):
            #To feed into the buffer
            dummy_reward = np.array([0.0])
            dummy_cost = np.array([0.0])
            dummy_state = np.zeros(self.state_shape)
            dummy_done = np.array([True])
            if(len(good_states) > 0):
                for good_state, good_action in zip(good_states, good_actions):
                    if(self.aug_state):
                        self.exp_good_buffer.append_roll(dummy_state, good_action, dummy_state, dummy_reward, dummy_cost, dummy_done, good_state)
                    else:
                        self.exp_good_buffer.append_roll(good_state, good_action, good_state, dummy_reward, dummy_cost, dummy_done)
                    
                    self.new_good += 1
            
            if(len(bad_states) > 0):
                for bad_state, bad_action in zip(bad_states, bad_actions):
                    if(self.aug_state):
                        self.exp_bad_buffer.append_roll(dummy_state, bad_action, dummy_state, dummy_reward, dummy_cost, dummy_done, bad_state)
                    else:
                        self.exp_bad_buffer.append_roll(bad_state, bad_action, bad_state, dummy_reward, dummy_cost, dummy_done)
                    
                    self.new_bad += 1

            if(self.hash_map is not None):
                batch_states, batch_actions = None, None
                if(len(good_states) == 0):
                    batch_states = np.array(bad_states)
                    batch_actions = np.array(bad_actions)
                    
                elif(len(bad_states) == 0):
                    batch_states = np.array(good_states)
                    batch_actions = np.array(good_actions)
                    
                else:
                    batch_states = np.concatenate((np.array(good_states), np.array(bad_states)), axis=0)
                    batch_actions = np.concatenate((np.array(good_actions), np.array(bad_actions)), axis=0) 
                
                if(batch_states is not None):
                    self.hash_map.add(batch_states, batch_actions)
        
        threads_clfs = []
        for i in range(self.n_ensemble): 
            threads_clfs.append(threading.Thread(target=self.train_clf, args=(self.clfs[i], self.optim_clfs[i], self.epoch_clfs)))
            threads_clfs[-1].start()

        for i in range(self.n_ensemble):
            threads_clfs[i].join()

        _ = self.validate_clfs('in')
            
        if(self.aug_state):
            states, actions, env_rewards, _, _, dones, log_pis, next_states, aug_states = self.buffer.get()

        else:
            states, actions, env_rewards, _, _, dones, log_pis, next_states = self.buffer.get()
        
        rewards = env_rewards
        with torch.no_grad():
            if(self.exp_bad_buffer.roll_n>=self.start_bad and self.n_step>=self.warm_start_steps):
                costs_clfs = []
                for i in range(self.n_ensemble):
                    if(self.aug_state):
                        costs_clfs.append(torch.sigmoid(self.clfs[i](aug_states, actions)))
                    else:
                        costs_clfs.append(torch.sigmoid(self.clfs[i](states, actions)))
                costs_clfs = torch.stack(costs_clfs)
                classes = torch.zeros_like(costs_clfs, dtype=torch.bool, device=self.device)
                classes[costs_clfs>self.class_prob] = True 
                costs_clfs = torch.zeros_like(env_rewards, dtype=torch.float, device=self.device)
                costs_clfs[classes.sum(dim=0)>self.n_ensemble//2] = 1.0
                # costs_clfs[classes.sum(dim=0)==self.n_ensemble] = 1.0
                # costs_ent = classes.float().std(dim=0)
                # costs_clfs -= costs_ent*0.5

                # majority voting + exploration bonus
                # costs_clfs = torch.mean(costs_clfs, dim=0) - torch.std(costs_clfs, dim=0)
                costs_clfs = costs_clfs.clamp(min=0.0, max=1.0)

            else:
                costs_clfs = torch.zeros_like(env_rewards)
        
        self.update_ppo(
            states, actions, rewards, costs_clfs, dones, log_pis, next_states)
            
        print(f'[Train] R: {np.mean(self.return_reward):.2f}, C_gt: {np.mean(self.return_cost):.2f}, '+
              f'C_clfs: {costs_clfs.mean().item()*self.max_episode_length:.2f}, '+
              f'newB: {self.new_bad}, newG: {self.new_good} '+
              f'ratio: {ratio:.2f} '+
              f'Queries: {self.n_queries}, Trjs_queried: {self.n_trajs_queried}')
    
        
        if(self.wandb_log):
            wandb.log({'train/R': np.mean(self.return_reward), 'train/C': np.mean(self.return_cost),
                    'train/C_clfs': costs_clfs.mean().item()*self.max_episode_length, 'buffer/newB': self.new_bad, 'buffer/newG': self.new_good,
                    'train/l': self.penalty.item(), 'buffer/bad_buffer_size': self.exp_bad_buffer.roll_n, 
                    'buffer/good_buffer_size': self.exp_good_buffer.roll_n,'train/target_cost': self.target_cost*self.max_episode_length*(1-self.cost_gamma),
                    'buffer/query_ratio': ratio, 'buffer/n_queries': self.n_queries, 'buffer/n_trajs_queried': self.n_trajs_queried})
                    
        self.return_cost = []
        self.return_reward = []
        self.new_good = 0
        self.new_bad = 0
        self.novel_trajs = 0

    def train_clf(self, clf, optim, n_epochs):
        
        if(self.exp_bad_buffer.roll_n == 0 or self.exp_good_buffer.roll_n == 0):
            return
        
        for _ in range(n_epochs):

            if(self.over_sample):
                bad_states, bad_actions, good_states, good_actions = self.sample_state_actions(self.batch_size)
                n_good_sample = self.batch_size
                n_bad_sample = self.batch_size
            else:
                n_good = self.exp_good_buffer.roll_n
                n_bad = self.exp_bad_buffer.roll_n
                prop_good = n_good/(n_good+n_bad)
                n_good_sample = int(2*self.batch_size*prop_good)
                n_bad_sample = 2*self.batch_size - n_good_sample
                _, _, good_states, good_actions = self.sample_state_actions(n_good_sample)
                bad_states, bad_actions, _, _ = self.sample_state_actions(n_bad_sample)
            
            batch_states = torch.cat((good_states, bad_states), dim=0)
            batch_actions = torch.cat((good_actions, bad_actions), dim=0)
            batch_weights = torch.ones((2*self.batch_size, 1), dtype=torch.float32, device=self.device)
            labels = torch.cat((torch.zeros((n_good_sample,1), device=self.device, dtype=torch.float32), torch.ones((n_bad_sample,1), device=self.device, dtype=torch.float32)), dim=0)

            self.update_clfs(clf, optim, batch_states, batch_actions, labels, batch_weights, self.pos_weight)
    
    def update_clfs(self, 
                    clf, optim, batch_states, batch_actions, 
                    labels_bad_is_one, weight, pos_weight
                    ):
        optim.zero_grad()

        clf_logits = clf(batch_states, batch_actions)
        loss = F.binary_cross_entropy_with_logits(clf_logits, labels_bad_is_one, weight=weight, pos_weight=pos_weight)
        if(self.hinge_coeff>0):
            hl = hinge_loss(clf_logits, labels_bad_is_one)
            loss += self.hinge_coeff*hl
        loss.backward()
        nn.utils.clip_grad_norm_(clf.parameters(), self.max_grad_norm)
        self.grad_norm = self.calculate_gradient_norm(clf)
        optim.step()         

    def save_models(self, save_dir):
        super().save_models(save_dir)
        for i in range(self.n_ensemble):
            torch.save(self.clfs[i].state_dict(), f'{save_dir}/clfs{i}.pth')

    def sample_state_actions(self, batch_size=None):
        if(batch_size is None):
            batch_size = self.batch_size

        if(self.aug_state):
            _, bad_actions,_,_,_,_, bad_states = self.exp_bad_buffer.sample_roll(batch_size)
            _, good_actions,_,_,_,_, good_states = self.exp_good_buffer.sample_roll(batch_size)
        
        else:
            bad_states, bad_actions,_,_,_,_ = self.exp_bad_buffer.sample_roll(batch_size)
            good_states, good_actions,_,_,_,_ = self.exp_good_buffer.sample_roll(batch_size)

        return bad_states, bad_actions, good_states, good_actions
    
    
    def get_novelty(self, traj_states, traj_actions):
        if(isinstance(traj_states, list)):
            traj_states = np.array(traj_states)
            traj_actions = np.array(traj_actions)
        
        densities = self.hash_map.get_density(traj_states, traj_actions)
        novels = np.abs(densities) < 1e-12

        return novels.sum() > 0.0
    
    def validate_clfs(self, dist_type):

        with torch.no_grad():
            if(self.exp_bad_buffer.roll_n == 0 or self.exp_good_buffer.roll_n == 0):
                return
            
            if(self.over_sample):
                bad_states, bad_actions, good_states, good_actions = self.sample_state_actions(self.batch_size)
                n_good_sample = self.batch_size
                n_bad_sample = self.batch_size
                
            else:
                # sample good and bad in proportion to size of buffers
                n_good = self.exp_good_buffer.roll_n
                n_bad = self.exp_bad_buffer.roll_n
                n_bad_sample = int((n_bad) * (2*self.batch_size) / (n_good + n_bad) )
                n_good_sample = int(2*self.batch_size - n_bad_sample)
                bad_states, bad_actions, _, _ = self.sample_state_actions(n_bad_sample)
                _, _, good_states, good_actions = self.sample_state_actions(n_good_sample)             
            
                preds = []
                for i in range(self.n_ensemble):
                    bad_logits = self.clfs[i](bad_states, bad_actions)
                    good_logits = self.clfs[i](good_states, good_actions)
                    disc_logits = torch.cat((good_logits, bad_logits), dim=0)
                    #classify disc_logits
                    pred_ = torch.zeros_like(disc_logits, dtype=torch.bool, device=self.device)
                    pred_[disc_logits<0.0] = True
                    preds.append(pred_)

                preds = torch.stack(preds, dim=0)
                preds = torch.sum(preds, dim=0)
                    
                #majority voting
                preds_good = torch.zeros_like(preds, dtype=torch.bool, device=self.device)
                is_good = preds > self.n_ensemble//2
                preds_good[is_good] = True
                
                labels = torch.cat((torch.zeros((n_good_sample,1), device=self.device, dtype=torch.bool), torch.ones((n_bad_sample,1), device=self.device, dtype=torch.bool)), dim=0)

                train_stats = compute_train_stats(
                    preds_good, labels
                )

                if(self.wandb_log):
                    for k, v in train_stats.items():
                        wandb.log({f'disc/{dist_type}/'+k: v})

                for k, v in train_stats.items():
                    print(f'disc/{dist_type}/'+k, v)

            
    def calculate_gradient_norm(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        
        return total_norm
            
    
