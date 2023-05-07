from gym import Env
from gym.spaces.box import Box

class ResettableEnv(Env):
    """Abstract class for resettable Gym environments"""
    
    def reset(self, initial_state=None):
        """Pass in an initial state to reset the environment to that state. (This only works if the wrapper is in planning model mode)"""
        pass

    def separate_resettable_part(self, obs):
        """Separates the observation into the resettable portion and the original. Make sure this operation is differentiable"""
        return obs, obs

    def combine_resettable_part(self, obs, resettable):
        """Combines an observation that has been split like in separate_resettable_part back together. Make sure this operation is differentiable"""
        return resettable

    def sample_obs(self, **kwargs):
        """Automatically sample an observation to seed state generation"""
        return self.observation_space.sample()

    def resettable_bounds(self):
        """Get bounds for resettable part of observation space"""
        return self.observation_space.low, self.observation_space.high