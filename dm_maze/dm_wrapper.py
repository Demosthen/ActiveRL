from copy import copy
import imp
from typing import Any
from gym import spaces

from dm_control import suite
from dm_env import specs, TimeStep
from dm2gym.envs.dm_suite_env import DMSuiteEnv
import numpy as np

def convert_dm_control_to_gym_space(dm_control_space):
    r"""Convert dm_control space to gym space. """
    if isinstance(dm_control_space, specs.BoundedArray):
        if np.ndim(dm_control_space.minimum) == 0:
            dm_min = np.full(dm_control_space.shape, dm_control_space.minimum)
        else:
            dm_min = dm_control_space.minimum
        if np.ndim(dm_control_space.maximum) == 0:
            dm_max = np.full(dm_control_space.shape, dm_control_space.maximum)
        else:
            dm_max = dm_control_space.maximum
        space = spaces.Box(low=dm_min,
                           high=dm_max, 
                           dtype=dm_control_space.dtype,
                           shape=dm_control_space.shape)
        assert space.shape == dm_control_space.shape
        return space

    elif isinstance(dm_control_space, specs.Array) and not isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(low=-float('inf'), 
                           high=float('inf'), 
                           shape=dm_control_space.shape, 
                           dtype=dm_control_space.dtype)
        return space
    elif isinstance(dm_control_space, dict):
        space = spaces.Dict({key: convert_dm_control_to_gym_space(value)
                             for key, value in dm_control_space.items()})
        return space

class DMEnvWrapper(DMSuiteEnv):
    def __init__(self, config):
        config = copy(config)
        dm_env = config["dm_env"]
        del config["dm_env"]
        self.env = dm_env(**config)
        self.metadata = {'render.modes': ['human', 'rgb_array'],
                         'video.frames_per_second': round(1.0/self.env.control_timestep())}

        self.observation_space = convert_dm_control_to_gym_space(self.env.observation_spec())
        self.action_space = convert_dm_control_to_gym_space(self.env.action_spec())
        self.viewer = None

    def reset(self, initial_state=None):
        timestep = self.env.reset(initial_state)
        return timestep.observation

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.env, __name)
