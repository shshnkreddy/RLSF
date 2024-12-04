import gymnasium

class NavClassifierWrapper(gymnasium.Wrapper):

    def reset(self, seed=None):
        obs, info = super().reset(seed=seed)
        info['position'] = self.env.task.agent.pos[:2]
        info['velocity'] = self.env.task.agent.vel[:2]

        return obs, info

    def step(self, action):        
        obs, reward, cost, terminated, truncated, info = super().step(action)
        
        info['position'] = self.env.task.agent.pos[:2]
        info['velocity'] = self.env.task.agent.vel[:2]

        return obs, reward, cost, terminated, truncated, info
    
class VelClassifierWrapper(gymnasium.Wrapper):

    def reset(self, seed=None):
        obs, info = super().reset(seed=seed)
        x_position = 0.0
        x_velocity = 0.0
        y_position = 0.0
        y_velocity = 0.0

        info['position'] = [x_position, y_position]
        info['velocity'] = [x_velocity, y_velocity]

        return obs, info

    def step(self, action):        
        obs, reward, cost, terminated, truncated, info = super().step(action)
        
        x_position = info['x_position']
        x_velocity = info['x_velocity']
        
        y_position = 0.0
        y_velocity = 0.0
        if('y_velocity' in info):
            y_velocity = info['y_velocity']
        if('y_position' in info):
            y_position = info['y_position']

        info['position'] = [x_position, y_position]
        info['velocity'] = [x_velocity, y_velocity]

        return obs, reward, cost, terminated, truncated, info
    

    




