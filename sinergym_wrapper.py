import gym

import numpy as np
from ray.rllib.utils.numpy import one_hot
import torch
from gym_simplegrid.envs.simple_grid import SimpleGridEnvRLLib
from gym.spaces.box import Box
from gym.envs.toy_text.utils import categorical_sample
from resettable_env import ResettableEnv
import sinergym

class SynergymWrapper(gym.core.ObservationWrapper, ResettableEnv):

    def __init__(self, config):
        env = gym.make('Eplus-5Zone-hot-discrete-stochastic-v1')
        env.weather_variability = config["weather_variability"]
        super().__init__(env)
        self.env = env

        # Augment observation space with weather variability info
        obs_space = env.observation_space
        obs_space_shape_list = list(obs_space.shape)
        obs_space_shape_list[-1] += 3

        low = list(obs_space.low) + [-5e5, -5e5, -5e5]
        high = list(obs_space.high) + [5e5, 5e5, 5e5]
        self.observation_space = Box(
            low = np.array(low), 
            high = np.array(high),
            shape = obs_space_shape_list,
            dtype=np.float32)
        self.is_evaluation = config["is_evaluation"]

    def observation(self, observation):
        variability = np.array(self.env.weather_variability)
        return np.concatenate([observation, variability], axis=-1)

    def inverse_observation(self, observation):
        return observation[..., :-3]

    def separate_resettable_part(self, obs):
        """Separates the observation into the resettable portion and the original. Make sure this operation is differentiable"""
        if obs is None:
            return self.env.weather_variability
        return obs[..., -3:], obs

    def combine_resettable_part(self, obs, resettable):
        """Combines an observation that has been split like in separate_resettable_part back together. Make sure this operation is differentiable"""
        # Make sure torch doesn't backprop into non-resettable part
        obs = obs.detach()
        obs[..., -3:] = resettable
        return obs

    def resettable_bounds(self):
        """Get bounds for resettable part of observation space"""
        low = np.array([-1., 0., 0.])
        high = np.array([1., 1., 5.])
        return low, high

    def reset(self, initial_state=None):
        obs = self.env.reset()
        if initial_state is not None:
            _, obs, _ = self.env.simulator.reset(self.separate_resettable_part(initial_state)[0])
            obs = np.array(obs, dtype=np.float32)
        return self.observation(obs)

    
    
    
