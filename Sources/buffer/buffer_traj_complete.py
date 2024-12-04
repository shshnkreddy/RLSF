import numpy as np

class Schedule:
    def __init__(self, n_samples_rollout, total_traj_queries, max_episode_length, total_timesteps, schedule='uniform'):
        self.t = 0
        # total_timesteps = min(int(5*1e6), total_timesteps)
        self.n_samples_rollout = n_samples_rollout
        self.schedule = schedule
        self.max_episode_length = max_episode_length
        self.total_traj_queries = total_traj_queries

        self.trajs_per_rollout = n_samples_rollout//max_episode_length
        self.total_rollout_trajs = total_timesteps//max_episode_length
        self.n_updates = total_timesteps/n_samples_rollout

        self.n_queries = 0

        if(self.schedule=='decreasing'):
            #Schedule is (c^t*n_samples_rollout)
            # self._c = 1-(self.total_traj_queries/(self.total_rollout_trajs)) + 0.005
            #Decrease c proportiinal to t/t+T
            ts = np.arange(self.n_updates)
            T = self.n_updates
            self._c = (total_traj_queries/self.trajs_per_rollout)/(np.sum(1-2*ts/(ts+T)))

        else:
            self._nq = self.total_traj_queries/self.n_updates

    def step(self):
        if(self.schedule=='uniform'):
            n_q = self._nq
        elif(self.schedule=='decreasing'):
            # n_q = (self._c**self.t) * self.trajs_per_rollout
            n_q = self._c * (1-2*self.t/(self.t+self.n_updates)) * self.trajs_per_rollout
        
        if(n_q < 1 and self.n_queries < self.total_traj_queries):
            n_q = 1
            
        if(self.n_queries >= self.total_traj_queries):
            n_q = 0
    
        n_q = int(n_q)
        c = n_q/self.total_traj_queries
        self.t += 1
        self.n_queries += n_q
        return int(n_q), c

class Trajectory_Buffer_Query:
    #Store entire trajectory
    def __init__(self, segment_length, env_cost_limit, state_shape, action_shape, scheduler=None):
        self.trajs = []
        self.segment_length = segment_length
        self.env_cost_limit = env_cost_limit  
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.scheduler = scheduler

    def add(self, states, actions, rewards, costs, entropy=None, novel=None):
        #Check shapes of states and actions
        assert states[0].shape == self.state_shape, "State shape mismatch"
        assert actions[0].shape == self.action_shape, "Action shape mismatch"
        traj = {}
        traj['states'] = states
        traj['actions'] = actions
        # traj['aux'] = aux
        traj['rewards'] = rewards
        traj['costs'] = costs
        traj['entropy'] = entropy
        traj['novel'] = novel
        self.trajs.append(traj)

    def reset(self):
        self.trajs = []

    def get(self):
        return self.trajs
    
    def query_user(self, strat='all'):
        query_trajs, ratio = self.filter_trajs(strat=strat)
        good_states = []
        good_actions = []
        bad_states = []
        bad_actions = []
        n_queries = 0
        n_trajs_queried = 0

        for traj in query_trajs:
            states = traj['states']
            actions = traj['actions']
            costs = traj['costs']

            # Check if it is a good trajectory
            total_cost = np.sum(costs)
            if(total_cost <= 0.0):
                good_states.extend(states)
                good_actions.extend(actions)
                n_queries += 1
                n_trajs_queried += 1
            
            else:
                # Segment cost comparison
                tmp_return_cost_cum = np.cumsum(costs)
                bads = np.zeros(len(tmp_return_cost_cum), dtype=bool)

                _segment_length = min(self.segment_length, len(tmp_return_cost_cum))
                _n_queries = 0
                for i in range(0, len(tmp_return_cost_cum), _segment_length):
                    segment_cost = tmp_return_cost_cum[i+_segment_length-1] - tmp_return_cost_cum[i] + costs[i]
                    if(segment_cost > self.env_cost_limit):
                        bads[i:i+_segment_length] = True
                    _n_queries += 1

                for state, action, i in zip(states, actions, range(len(states))):
                    if(bads[i]):
                        bad_states.append(state)
                        bad_actions.append(action)
                    else:
                        good_states.append(state)
                        good_actions.append(action)

                n_trajs_queried += 1
                n_queries += _n_queries

        return good_states, good_actions, bad_states, bad_actions, n_queries, n_trajs_queried, ratio


    def filter_trajs(self, strat):
        if(strat == 'novel'):
            #Filter novel trajs
            if(self.scheduler is not None):
                self.trajs = sorted(self.trajs, key=lambda x: x['novel'], reverse=True)
                n_queries, c = self.scheduler.step()
                n_queries = min(n_queries, len(self.trajs))
                query_trajs = self.trajs[:n_queries]
            else:
                query_trajs = [traj for traj in self.trajs if traj['novel'] == 1]
                c = len(query_trajs) / len(self.trajs)
        
        elif(strat == 'random'):
            n_queries, c = self.scheduler.step()
            n_queries = min(n_queries, len(self.trajs))
            query_trajs = np.random.choice(self.trajs, n_queries, replace=False)
        
        elif(strat == 'entropy'):
            self.trajs = sorted(self.trajs, key=lambda x: x['entropy'], reverse=True)

            n_queries, c = self.scheduler.step()
            n_queries = min(n_queries, len(self.trajs))
            query_trajs = self.trajs[:n_queries]

        elif(strat == 'all'):
            query_trajs = self.trajs
            c = 1

        else:
            raise ValueError('Invalid strategy')

        self.trajs = []

        return query_trajs, c

        


        



    
    
     
