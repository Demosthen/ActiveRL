from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Union
from webbrowser import get
from citylearn.citylearn import CityLearnEnv
from citylearn.building import Building
import gym
from gym import Env
import numpy as np
from ray.rllib.utils.numpy import one_hot
import torch
from citylearn.base import Environment
from citylearn.data import EnergySimulation, CarbonIntensity, Pricing, Weather
from citylearn.energy_model import Battery, ElectricHeater, HeatPump, PV, StorageTank
from gym.envs.toy_text.utils import categorical_sample
from citylearn_model_training.planning_model import get_planning_model
from gym.spaces.box import Box

class CityLearnEnvWrapper(gym.core.ObservationWrapper, gym.core.ActionWrapper, gym.core.RewardWrapper):
    """
        Wraps the SimpleGridEnv to make discrete observations one hot encoded. Also provides an inverse_observations
        function that can transform the one hot encoded observations back to the original discrete space.
        If you specify a planning_model_ckpt in the config function, will output state transitions from the planning model
        instead of the environment.

    """
    def __init__(self, config: Dict):
        # read in planning model ckpt path and whether this env is used for evaluation or not
        planning_model_ckpt = config["planning_model_ckpt"] if "planning_model_ckpt" in config else None
        self.is_evaluation = config["is_evaluation"]

        # Get config ready to pass into CityLearnEnv
        config = deepcopy(config)
        if "planning_model_ckpt" in config:
            del config["planning_model_ckpt"] # Citylearn will complain if you pass it this extra parameter
        del config["is_evaluation"]

        #Initialize CityLearnEnv
        env = CityLearnEnv(**config)
        super().__init__(env)
        self.env = env

        #Implement switch between planning model and citylearn
        if planning_model_ckpt is not None:
            self.planning_model = get_planning_model(planning_model_ckpt)
            self.planning_model.eval_batchnorm()
            self.observation_space = self.env.observation_space[0]
            self.action_space = self.env.action_space[0]
        else:
            self.planning_model = None
            self.observation_space = self.env.observation_space[0]
            self.action_space = self.env.action_space[0]

        # Bookkeeping to make sure we reset after the right number of timesteps
        self.curr_obs = self.reset()
        self.time_steps = self.env.time_steps
        self.time_step = 0

    # Override `observation` to custom process the original observation
    # coming from the env.
    def observation(self, observation):
        return observation[0]

    # Override `reward` to custom process the original reward
    # coming from the env.
    def reward(self, reward):
        return reward[0]

    # Override `action` to custom process the original action
    # coming from the policy.
    def action(self, action):
        return [action]

    def compute_reward(self, obs: Union[np.ndarray, torch.Tensor]):
        """
            Computes the reward from CityLearn given an observation.
            Painstakingly reverse engineered and tested against CityLearn's default reward behavior
        """
        electricity_consumption_index = 23
        carbon_emission_index = 19
        electricity_pricing_index = 24
        num_buildings = len(self.env.buildings)
        offset = (len(obs) - len(self.env.shared_observations)) // num_buildings
        net_electricity_consumptions = [obs[i] for i in range(electricity_consumption_index, electricity_consumption_index + offset * num_buildings, offset)]
        # NOTE: citylearn clips carbon emissions PER BUILDING but electricity cost IN AGGREGATE
        carbon_emission = (sum([max(obs[carbon_emission_index] * net_electricity_consumptions[j], 0) for j in range(num_buildings)]))
        price = sum([obs[i] * net_electricity_consumptions[j] for j, i in enumerate(range(electricity_pricing_index, electricity_pricing_index + offset * num_buildings, offset))])
        price = max(price, 0)
        return - (carbon_emission + price)

    def step(self, action):
        if self.planning_model is None:
            obs, rew, done, info = self.env.step(self.action(action))
            return self.observation(obs), self.reward(rew), done, info
        else:

            planning_input = np.atleast_2d(np.concatenate([self.curr_obs, action]))
            self.curr_obs = self.planning_model.forward_np(planning_input).flatten()
            rew = self.compute_reward(self.curr_obs)
            self.next_time_step()
            return self.curr_obs, rew, self.done, {}

    @property
    def done(self) -> bool:
        """Check if simulation has reached completion."""

        return self.time_step == self.time_steps - 1

    def next_time_step(self):
        self.time_step += 1

    # DEPRECATED
    def inverse_observation(self, wrapped_obs):
        # Handle torch arrays properly.
        if isinstance(wrapped_obs, torch.Tensor):
            wrapped_obs = wrapped_obs.numpy()
        return wrapped_obs

    """Pass in an initial state to reset the environment to that state. (This only works if the wrapper is in planning model mode)"""
    def reset(self, initial_state=None):
        
        self.time_step = 0
        if self.planning_model is None:
            return self.observation(self.env.reset())
        else:
            if initial_state is not None:
                self.curr_obs = initial_state
            else:
                self.curr_obs = self.observation(self.env.reset())
            return self.curr_obs
        