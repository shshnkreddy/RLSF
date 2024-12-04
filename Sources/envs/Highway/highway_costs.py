import gymnasium as gym
from highway_env.envs import HighwayEnv, TwoWayEnv
import numpy as np

class HighWayEnvCost(HighwayEnv):
    def __init__(self, config, render_mode=None, limit_v=30.0, max_time_steps=100):
        self.limit_v = limit_v
        super().__init__(config=config, render_mode=render_mode)
        self.last_x = None
        self.t = None
        self.max_time_steps = max_time_steps

    def reset(self, seed=None, options=None):
        obs, info = super().reset()
        self.last_x = self.vehicle.position[0]
        self.t = 0
        return obs, info
    
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        cost = self._cost(action)
        self.t += 1
        if(self.t >= self.max_time_steps):
            truncated = True
        return obs, reward, cost, done, truncated, info

    def _reward(self, action):
        rewards = super()._rewards(action)

        rewards.pop("high_speed_reward")

        #Reward for moving down the road
        r_position = (self.vehicle.position[0] - self.last_x)/10
        rewards["forward_reward"] = r_position
        
        self.last_x = self.vehicle.position[0]

        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        
        if(rewards["on_road_reward"] == 0):
            reward = -1
        # reward *= rewards["on_road_reward"]

        return reward
    
    def _cost(self, action):
        #velocity cost
        v = self.vehicle.velocity[0]
        cost = 0
        if v > self.limit_v:
            cost += 1
        return 0

    
