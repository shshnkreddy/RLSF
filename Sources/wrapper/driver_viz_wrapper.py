import gymnasium
import numpy as np
from safety_gymnasium.wrappers import SafetyGymnasium2Gymnasium
from Sources.envs.Driver.render import DriverRender


class DriverVizWrapper(gymnasium.Wrapper):
    def __init__(self, env, resolution=(320,64), action_range=(-1.0, 1.0), stack_size=1):
        super(DriverVizWrapper, self).__init__(env)
        self.resolution = resolution
        #observation space is an image of dimension resolution
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(3*stack_size, resolution[0], resolution[1]), dtype=np.uint8)
        self.renderer = DriverRender(env, resolution[0], resolution[1])
        self.action_range = action_range

        self.stack_size = stack_size
        self.stack = np.zeros((stack_size, 3, resolution[0], resolution[1]), dtype=np.uint8)
        

    def _rescale_actions(self, action):
        action = [action[0], np.interp(action[1], (-1.0, 1.0), self.action_range)]
        return action

    def _gen_obs(self):
        if(self.stack_size>1):
            self.stack[:-1] = self.stack[1:]
            self.stack[-1] = self.renderer.render(mode='rgb_array').transpose(2,0,1)
            return self.stack.reshape(3*self.stack_size, self.resolution[0], self.resolution[1])

        return self.renderer.render(mode='rgb_array').transpose(2,0,1)
    
    def reset(self):
        _, info = self.env.reset()
        return self._gen_obs(), info
    
    def step(self, action):
        action = self._rescale_actions(action)
        obs, reward, cost, done, truncated, info = self.env.step(action)
        return self._gen_obs(), reward, cost, done, truncated, info
    
    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)
    
    def close(self):
        self.renderer.close()
        self.env.close()


    