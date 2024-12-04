import gymnasium

class BlockedWalkerWrapper(gymnasium.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        x_position = self.data.qpos[0:2].copy()[0]
        cost = self._calculate_cost(x_position)
        return obs, reward, cost, terminated, truncated, info
    
    def _calculate_cost(self, x_position):
        #cost=1 if x less than 3
        if x_position <= -3.0:
            return 1.0

        return 0.0
    
    def render(mode):
        return super.render(mode=mode)