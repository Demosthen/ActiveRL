import gym

import numpy as np
from ray.rllib.utils.numpy import one_hot
import torch
from gym_simplegrid.envs.simple_grid import SimpleGridEnvRLLib
from gym.spaces.box import Box
from gym.envs.toy_text.utils import categorical_sample
from resettable_env import ResettableEnv
import sinergym
import time
import os
from sinergym.utils.controllers import RBC5Zone, RBCDatacenter, RandomController

class SynergymWrapper(gym.core.ObservationWrapper, ResettableEnv):

    def __init__(self, config):
        # Waits a process-specific amount of time so as to avoid a race condition
        sleep_time = (os.getpid() % 10) / 10
        print(sleep_time)
        time.sleep(sleep_time)
        self.env_name = 'Eplus-5Zone-hot-discrete-stochastic-v1'
        env = gym.make(self.env_name)
        self.weather_variability = config["weather_variability"]
        self.scenario_idx = 0
        env.weather_variability = self.weather_variability[self.scenario_idx]
        super().__init__(env)
        self.env = env
        # Augment observation space with weather variability info
        obs_space = env.observation_space
        obs_space_shape_list = list(obs_space.shape)
        obs_space_shape_list[-1] += 3
        low = list(obs_space.low) + [-1., 0., 0.]
        high = list(obs_space.high) + [1., 1., 5.]
        self.observation_space = Box(
            low = np.array(low), 
            high = np.array(high),
            shape = obs_space_shape_list,
            dtype=np.float32)
        self.is_evaluation = config["is_evaluation"]
        self.last_untransformed_obs = None
        if config["use_rbc"]:
            if "5Zone" in self.env_name:
                self.replacement_controller = RBC5Zone(self.env)
            else:
                self.replacement_controller = RBCDatacenter(self.env)
        elif config["use_random"]:
            self.replacement_controller = RandomController(self.env)
        else:
            self.replacement_controller = None

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
        low = np.array([0., -20., 0.])
        high = np.array([10., 20., 5.])
        return low, high

    def reset(self, initial_state=None):
        obs = self.env.reset()
        self.last_untransformed_obs = obs
        if initial_state is not None:
            # Reset simulator with specified weather variability
            print(self.separate_resettable_part(initial_state)[0])
            _, obs, _ = self.env.simulator.reset(tuple(self.separate_resettable_part(initial_state)[0]))
            obs = np.array(obs, dtype=np.float32)
        else:
            # Cycle through the weather variability scenarios
            curr_weather_variability = self.weather_variability[self.scenario_idx]
            print(curr_weather_variability)
            self.env.simulator.reset(curr_weather_variability)
            self.scenario_idx = (self.scenario_idx + 1) % len(self.weather_variability)
        return self.observation(obs)

    def step(self, action):
        """Returns modified observations and inputs modified actions"""
        action = self.replace_action(self.last_untransformed_obs, action)
        print("ACTION GIVEN AT THIS STEP ISSSSSS", action)
        obs, reward, done, info = self.env.step(action)
        self.last_untransformed_obs = obs
        return self.observation(obs), reward, done, info

    def replace_action(self, obs, action):
        if self.replacement_controller is None:
            return action
        elif isinstance(self.replacement_controller, RandomController):
            print("ACTION REPLACEEDDDDEEDED3EDE")
            return self.replacement_controller.act()
        else:
            return self.replacement_controller.act(obs)
    
    
    
