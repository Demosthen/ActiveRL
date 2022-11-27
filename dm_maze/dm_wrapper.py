from collections import OrderedDict
from copy import copy, deepcopy
import imp
from typing import Any
from gym import Env, spaces

from dm_control import suite
from dm_env import specs, TimeStep
from dm2gym.envs.dm_suite_env import DMSuiteEnv
import gym
import numpy as np

import labmaze
from dm_maze.dm_maze import DM_Maze_Env, DM_Maze_Task, DM_Maze_Arena
from dm_control.locomotion.walkers import ant
from resettable_env import ResettableEnv
import torch

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


class DM_Maze_Wrapper(DMSuiteEnv):
    def __init__(self, config):
        config = self.process_config(config)

        self.initialize_env(config)

        self.metadata = {'render.modes': ['human', 'rgb_array'],
                         'video.frames_per_second': round(1.0/self.env.control_timestep())}

        self.observation_space = convert_dm_control_to_gym_space(
            self.env.observation_spec())
        self.action_space = convert_dm_control_to_gym_space(
            self.env.action_spec())
        self.viewer = None

    def initialize_env(self, config):
        walker = ant.Ant()
        arena = DM_Maze_Arena(
            maze=labmaze.FixedMazeWithRandomGoals(self.maze_str))
        task = DM_Maze_Task(walker, None, arena, arena.num_targets, contact_termination=True, aliveness_reward=self.aliveness_reward, use_all_geoms=self.use_all_geoms,
                            enable_global_task_observables=True, distance_reward_scale=self.distance_reward_scale, subtarget_rews=self.subtarget_rews)
        self.env = DM_Maze_Env(task=task, **config)

    def process_config(self, config):
        config = deepcopy(config)

        # Read extra (not from DM_Maze_Env) config arguments
        self.maze_str = config["maze_str"]
        self.subtarget_rews = config["subtarget_rews"]
        self.aliveness_reward = config["aliveness_reward"]
        self.distance_reward_scale = config["distance_reward_scale"]
        self.use_all_geoms = config["use_all_geoms"]

        del config["maze_str"]
        del config["subtarget_rews"]
        del config["aliveness_reward"]
        del config["distance_reward_scale"]
        del config["use_all_geoms"]

        return config

    def reset(self, initial_state=None):
        timestep = self.env.reset(initial_state)
        return timestep.observation

    def render(self, mode=None):
        pixels = []
        for camera_id in range(3):
            pixels.append(self.physics.render(camera_id=camera_id, width=240))
        return np.hstack(pixels)

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.env, __name)

    def seed(self, seed):
        return self.env.random_state.seed(seed)

class DM_Maze_Obs_Wrapper(gym.ObservationWrapper, ResettableEnv):
    def __init__(self, config: dict):
        env: Env = DM_Maze_Wrapper(config)
        super().__init__(env)
        obs_space = self.env.observation_space
        flat_mapping = OrderedDict({})
        low = []
        high = []
        curr_idx = 0
        total_space = {}
        for key, space in obs_space.items():
            space: spaces.Box = space
            if len(space.shape) < 3:
                space_len = int(np.prod(space.shape))
                flat_mapping[key] = (curr_idx, curr_idx + space_len)
                low.append(space.low.reshape([space_len]))
                high.append(space.high.reshape([space_len]))
                curr_idx += space_len
            else:
                total_space[key] = space

        low = np.concatenate(low)
        high = np.concatenate(high)
        self.flat_mapping = flat_mapping
        total_space["flat"] = spaces.Box(low, high, shape=(curr_idx,))
        self.observation_space = spaces.Dict(total_space)

    def observation(self, observation):
        output = OrderedDict({})
        flat = np.zeros(self.observation_space["flat"].shape)
        for key, value in observation.items():
            if len(value.shape) < 3:
                start, end = self.flat_mapping[key]
                flat[start:end] = value.flatten()
            else:
                output[key] = value
        output["flat"] = flat
        return output

    def inverse_observation(self, observation):
        if observation is None:
            return None
        output = OrderedDict({})
        for k, space in self.env.observation_space.items():
            if k in self.flat_mapping:
                idxs = self.flat_mapping[k]
                output[k] = observation["flat"][..., idxs[0]:idxs[1]].reshape(space.shape)
            else:
                output[k] = observation[k]
        return output

    def separate_resettable_part(self, obs):
        """Separates the observation into the resettable portion and non-resettable portion"""
        return obs["flat"][..., 9:12], obs

    def combine_resettable_part(self, obs, resettable):
        """Combines an observation that has been split like in separate_resettable_part back together"""
        if isinstance(obs["flat"], torch.Tensor):
            # Make sure torch doesn't backprop into non-resettable part
            obs["flat"] = obs["flat"].detach()
        obs["flat"][..., 9:12] = resettable
        return obs

    def resettable_bounds(self):
        """Get bounds for resettable part of observation space"""
        maze_width = self.task._maze_arena.maze.width
        maze_height = self.task._maze_arena.maze.height
        xy_scale = self.task._maze_arena.xy_scale
        x_offset = xy_scale * (maze_width - 1) / 2
        y_offset = xy_scale * (maze_height - 1) / 2
        low = np.array([-x_offset, -y_offset, 0])
        high = np.array([x_offset, y_offset, 1])
        return low, high

    def reset(self, initial_state=None):
        obs = self.env.reset(self.inverse_observation(initial_state))
        return self.observation(obs)