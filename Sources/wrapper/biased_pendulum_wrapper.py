import gymnasium

class BiasedPendulumWrapper(gymnasium.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        x_position = self.data.qpos[0:2].copy()[0]
        reward = self._calculate_reward(x_position)
        cost = self._calculate_cost(x_position)
        return obs, reward, cost, terminated, truncated, info
    
    def _calculate_reward(self, x_position):
        #reward = 0.1 if x >= 0
        if(x_position >= 0):
            return 0.1
        
        #reward = 1.0 of x < -0.01
        elif(x_position < -0.01):
            return 1.0
        
        #reward monotonically decreases from 1.0 to 0.1 as you move from -0.01 to 0
        return 1.0 - 0.9*(0.01+x_position)/0.01

    def _calculate_cost(self, x_position):
        #cost=1 if x less than -0.015
        if x_position <= -0.015:
            return 1.0

        return 0.0
    

        

