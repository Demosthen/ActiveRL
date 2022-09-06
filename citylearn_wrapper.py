from pathlib import Path
from typing import Any, Mapping, Union
from citylearn.citylearn import CityLearnEnv
import gym
from gym import Env
import numpy as np
from ray.rllib.utils.numpy import one_hot
import torch
from gym.envs.toy_text.utils import categorical_sample

class  CityLearnEnvWrapper(gym.core.ObservationWrapper):
    """
        Wraps the SimpleGridEnv to make discrete observations one hot encoded. Also provides an inverse_observations
        function that can transform the one hot encoded observations back to the original discrete space.

    """
    def __init__(self, config):
        env = CityLearnEnv(**config)
        super().__init__(env)
        self.env = env
        self.orig_obs_space = self.env.observation_space
        #obs_space = env.observation_space
        #self.obs_space_max = obs_space.n
        #self.observation_space = Box(0, 1, [self.obs_space_max])

    # Override `observation` to custom process the original observation
    # coming from the env.
    # TODO
    def observation(self, observation):
        pass

    #TODO
    def inverse_observation(self, wrapped_obs):
        # Handle torch arrays properly.
        if isinstance(wrapped_obs, torch.Tensor):
            wrapped_obs = wrapped_obs.numpy()
        pass

    def reset(self, initial_state=None):
        if initial_state is not None:
            initial_state = self.inverse_observation(initial_state)
        return self.observation(self.env.reset(initial_state=initial_state))

# class CityLearnEnvSBWrapper(Env):

#     def __init__(self, schema: Union[str, Path, Mapping[str, Any]], **kwargs):
#         self.env = CityLearnEnv(schema, **kwargs)
#         self.observation_space = self.env.observation_space[0]
#         self.action_space = self.env.action_space[0]

#     def reset(self):
#         print(self.env.reset()[0], self.observation_space)
#         return np.array(self.env.reset()[0])

#     def step(self, action):
#         return self.env.step([action])

#     def render(self, mode="human"):
#         return self.env.render()

#     def close(self):
#         return self.env.close()