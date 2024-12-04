import gymnasium

class DummyCostWrapper(gymnasium.Wrapper):
    def step(self, action):        
        obs, reward, terminated, truncated, info = super().step(action)
        info['cost'] = 0
        return obs, reward, terminated, truncated, info