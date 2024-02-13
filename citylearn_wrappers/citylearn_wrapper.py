from copy import deepcopy
from typing import Dict, Union
from citylearn.citylearn import CityLearnEnv
import gym
import numpy as np
import torch
from citylearn_wrappers.citylearn_model_training.planning_model import (
    get_planning_model,
)
from citylearn_wrappers.rbc_agent import RBCAgent
from core.resettable_env import ResettableEnv


class CityLearnEnvWrapper(
    gym.core.ObservationWrapper,
    gym.core.ActionWrapper,
    gym.core.RewardWrapper,
    ResettableEnv,
):
    """
    Wraps the CityLearnEnv, provides an interface to do state transitions from a planning model or from multiple CityLearn environments.
    If you specify a planning_model_ckpt in the config function, will output state transitions from the planning model
    instead of the environment. Can operate as multiple different environments if a list of schema paths is provided
    in config["schema"] instead of a single path.
    """

    def __init__(self, config: Dict):

        config = self.process_config(config)

        self.initialize_subenvs(config)

        # Makes sure __get_attr__ and other functions are overrided by gym Wrapper for current env
        super().__init__(self.env)

        self.initialize_planning_model(self.planning_model_ckpt)

        self.observation_space = self.env.observation_space[0]
        self.action_space = self.env.action_space[0]

        self.initialize_rbc(self.action_space)

        # Bookkeeping to make sure we reset after the right number of timesteps
        self.curr_env_idx = 0
        self.curr_obs = self.reset()
        self.time_steps = self.env.time_steps
        self.time_step = 0

    def process_config(self, config):

        # Read extra config arguments
        # read in planning model ckpt path and whether this env is used for evaluation or not
        self.planning_model_ckpt = (
            config["planning_model_ckpt"] if "planning_model_ckpt" in config else None
        )
        self.is_evaluation = config["is_evaluation"]
        self.use_rbc_residual = config["use_rbc_residual"]
        self.action_multiplier = config["action_multiplier"]

        # Get config ready to pass into CityLearnEnv# Get config ready to pass into CityLearnEnv
        config = deepcopy(config)
        if "planning_model_ckpt" in config:
            del config[
                "planning_model_ckpt"
            ]  # Citylearn will complain if you pass it this extra parameter
        del config["is_evaluation"]
        del config["use_rbc_residual"]
        del config["action_multiplier"]

        return config

    def initialize_subenvs(self, config):
        if isinstance(config["schema"], list):
            configs = [deepcopy(config) for _ in config["schema"]]
            for i, schema in enumerate(config["schema"]):
                configs[i]["schema"] = schema
            self.envs = [CityLearnEnv(**subconfig) for subconfig in configs]
            print(
                "EVAL ENV SCHEMA: ", [env.schema["root_directory"] for env in self.envs]
            )
        else:
            # Initialize CityLearnEnv
            self.envs = [CityLearnEnv(**config)]
            print(
                "TRAIN ENV SCHEMA: ",
                [env.schema["root_directory"] for env in self.envs],
            )
        self.env = self.envs[0]

    def initialize_rbc(self, action_space):
        self.rbc = RBCAgent(action_space)

    def initialize_planning_model(self, planning_model_ckpt):
        # Implement switch between planning model and citylearn
        if planning_model_ckpt is not None:
            self.planning_model = get_planning_model(planning_model_ckpt)
            self.planning_model.eval_batchnorm()
        else:
            self.planning_model = None

    # Override `observation` to custom process the original observation
    # coming from the env.
    def observation(self, observation):
        return observation[0]

    # Override `reward` to custom process the original reward
    # coming from the env.
    def reward(self, reward):
        return reward[0] / 100

    # Override `action` to custom process the original action
    # coming from the policy.
    def action(self, observation, action):
        if self.use_rbc_residual:
            rbc_action = self.rbc.compute_action(observation)
            # action = rbc_action
            action = self.action_multiplier * action + rbc_action
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
        net_electricity_consumptions = [
            obs[i]
            for i in range(
                electricity_consumption_index,
                electricity_consumption_index + offset * num_buildings,
                offset,
            )
        ]
        # NOTE: citylearn clips carbon emissions PER BUILDING but electricity cost IN AGGREGATE
        carbon_emission = sum(
            [
                max(obs[carbon_emission_index] * net_electricity_consumptions[j], 0)
                for j in range(num_buildings)
            ]
        )
        price = sum(
            [
                obs[i] * net_electricity_consumptions[j]
                for j, i in enumerate(
                    range(
                        electricity_pricing_index,
                        electricity_pricing_index + offset * num_buildings,
                        offset,
                    )
                )
            ]
        )
        price = max(price, 0)
        return -(carbon_emission + price)

    def step(self, action):
        if self.planning_model is None or self.is_evaluation:
            obs, rew, done, info = self.env.step(self.action(self.curr_obs, action))
            return self.observation(obs), self.reward(rew), done, info
        else:
            planning_input = np.atleast_2d(
                np.concatenate([self.curr_obs, self.action(action)[0]])
            )
            self.curr_obs = self.planning_model.forward_np(planning_input).flatten()
            rew = self.compute_reward(
                self.curr_obs
            )  # - self.planning_model.compute_uncertainty(planning_input)
            self.next_time_step()
            return self.observation([self.curr_obs]), self.reward([rew]), self.done, {}

    @property
    def done(self) -> bool:
        """Check if simulation has reached completion."""

        return self.time_step == self.time_steps - 1

    def next_time_step(self):
        """Increments current timestep"""
        self.time_step += 1

    def reset_time_step(self):
        """Increments current timestep to 0"""
        self.time_step = 0

    def next_env(self):
        """Sets current environment to next environment in self.envs list"""
        self.curr_env_idx = (self.curr_env_idx + 1) % len(self.envs)
        self.env = self.envs[self.curr_env_idx]
        print("SWAPPING ENV TO ", self.env.schema["root_directory"])
        # Makes sure __get_attr__ and other functions are overrided by gym Wrapper for current env
        # super().__init__(self.env)

    # DEPRECATED
    def inverse_observation(self, wrapped_obs):
        # Handle torch arrays properly.
        if isinstance(wrapped_obs, torch.Tensor):
            wrapped_obs = wrapped_obs.numpy()
        return wrapped_obs

    """Pass in an initial state to reset the environment to that state. (This only works if the wrapper is in planning model mode)"""

    def reset(self, initial_state=None):

        self.reset_time_step()
        if self.is_evaluation:
            return self.observation(self.env.reset())
        elif initial_state is not None:
            self.curr_obs = initial_state
        else:
            self.curr_obs = self.observation(self.env.reset())
        return self.curr_obs

    def separate_resettable_part(self, obs):
        """Separates the observation into the resettable portion and non-resettable portion"""
        return obs, obs

    def combine_resettable_part(self, obs, resettable):
        """Combines an observation that has been split like in separate_resettable_part back together"""
        return resettable

    def resettable_bounds(self):
        """Get bounds for resettable part of observation space"""
        return self.observation_space.low, self.observation_space.high
