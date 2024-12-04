import gymnasium as gym
import numpy as np   

class BirdEyeWrapper(gym.ObservationWrapper):
    def __init__(self, env, normalize=True):
        super(BirdEyeWrapper, self).__init__(env)
        self.normalize = normalize
        if(normalize):
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(3, 64, 64),
                dtype=np.float32
            )
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, 64, 64),
            dtype=np.uint8
        )
    
    def observation(self, obs):
        if(self.normalize):
            return obs['birdeye'].transpose(2, 0, 1) / 255.0
        return obs['birdeye'].transpose(2, 0, 1)
    
class CameraWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(CameraWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(64, 64, 3),
            dtype=np.uint8
        )
    
    def observation(self, obs):
        return obs['camera']
    
    
class DummyVecEnv:
    def __init__(self, env):
        self.envs = [env]
        self.num_envs = 1
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, seed=None):
        obs, info = self.envs[0].reset()
        return [obs], [info]
    
    def step(self, actions):
        action = actions[0]
        obs, r, c, done, truncated, info = self.envs[0].step(action)
        if(done):
            obs, info = self.envs[0].reset()
        return [obs], [r], [c], [done], [truncated], [info]
    
    def close(self):
        return self.envs[0].close()
    
    def render(self, mode=None):
        return [self.envs[0].render()]