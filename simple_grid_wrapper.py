from mimetypes import init
from textwrap import wrap
from typing import Callable
import gym
import numpy as np
from ray.rllib.utils.numpy import one_hot
import torch
from gym_simplegrid.envs.simple_grid import SimpleGridEnvRLLib
from gym.spaces.box import Box
from gym.envs.toy_text.utils import categorical_sample

class SimpleGridEnvWrapper(gym.core.ObservationWrapper):
    """
        Wraps the SimpleGridEnv to make discrete observations one hot encoded. Also provides an inverse_observations
        function that can transform the one hot encoded observations back to the original discrete space.

    """
    def __init__(self, config):
        env = SimpleGridEnvRLLib(config)
        super().__init__(env)
        self.env = env
        obs_space = env.observation_space
        self.obs_space_max = obs_space.n
        self.observation_space = Box(0, 1, [self.obs_space_max])

    # Override `observation` to custom process the original observation
    # coming from the env.
    def observation(self, observation):
        # E.g. one-hotting a float obs [0.0, max_obs_space]
        return one_hot(observation, depth=self.obs_space_max)

    def project(self, obs):
        """
            Function added for ActiveRL:
            projects the observation onto the observation space
        """
        int_obs = torch.round(obs).int()
        if int_obs < 0:
            return torch.tensor([categorical_sample(self.initial_state_distrib, self.np_random)])
        return int_obs

    def inverse_observation(self, wrapped_obs):
        # Handle torch arrays properly.
        if isinstance(wrapped_obs, torch.Tensor):
            wrapped_obs = wrapped_obs.numpy()
        return np.argmax(wrapped_obs)

    def reset(self, initial_state=None):
        if initial_state is not None:
            initial_state = self.inverse_observation(initial_state)
        return self.observation(self.env.reset(initial_state=initial_state))
