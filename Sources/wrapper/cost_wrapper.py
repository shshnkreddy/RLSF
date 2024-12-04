import gymnasium

class CostWrapper(gymnasium.Wrapper):

    def step(self, action):        
        obs, reward, cost, terminated, truncated, info = super().step(action)
        cost = float(cost>0)
        return obs, reward, cost, terminated, truncated, info
    
    

    




