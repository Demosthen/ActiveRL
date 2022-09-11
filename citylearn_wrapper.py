from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Union
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

class CityLearnEnvWrapper(gym.core.ObservationWrapper, gym.core.ActionWrapper, gym.core.RewardWrapper):
    """
        Wraps the SimpleGridEnv to make discrete observations one hot encoded. Also provides an inverse_observations
        function that can transform the one hot encoded observations back to the original discrete space.
        If you specify a planning_model_ckpt in the config function, will output state transitions from the planning model
        instead of the environment.

    """
    def __init__(self, config):
        planning_model_ckpt = config["planning_model_ckpt"] if "planning_model_ckpt" in config else None
        if planning_model_ckpt is not None:
            self.planning_model = get_planning_model(planning_model_ckpt)
        else:
            self.planning_model = None
        config = deepcopy(config)
        if "planning_model_ckpt" in config:
            del config["planning_model_ckpt"] # Citylearn will complain if you pass it this extra parameter
        env = CityLearnEnv(**config)
        super().__init__(env)
        self.env = env
        #obs_space = env.observation_space
        #self.obs_space_max = obs_space.n
        self.observation_space = self.env.observation_space[0]
        self.action_space = self.env.action_space[0]
        self.curr_obs = self.env.reset()


    # Override `observation` to custom process the original observation
    # coming from the env.
    # TODO
    def observation(self, observation):
        return observation[0]

    def reward(self, reward):
        return reward[0]

    def action(self, action):
        return [action]

    def compute_reward(self, obs):
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
        carbon_emission = (sum([np.clip(obs[carbon_emission_index] * net_electricity_consumptions[j], 0, None) for j in range(num_buildings)]))
        price = sum([obs[i] * net_electricity_consumptions[j] for j, i in enumerate(range(electricity_pricing_index, electricity_pricing_index + offset * num_buildings, offset))])
        price = np.clip(price, 0, None)
        return - (carbon_emission + price)

    def step(self, action):
        if self.planning_model is None:
            obs, rew, done, info = self.env.step(self.action(action))

            return self.observation(obs), self.reward(rew), done, info
        else:
            planning_input = np.concatenate([self.curr_obs, action])
            next_obs = self.planning_model(planning_input)
            rew = self.compute_reward(next_obs)

    #TODO
    def inverse_observation(self, wrapped_obs):
        # Handle torch arrays properly.
        if isinstance(wrapped_obs, torch.Tensor):
            wrapped_obs = wrapped_obs.numpy()
        return wrapped_obs

    """Hoo boy this is going to get really involved"""
    def reset(self, initial_state=None):
        if initial_state is not None:
            initial_state = self.inverse_observation(initial_state)

        # for building in self.env.buildings:
        #     self.building_reset(building, initial_state=initial_state)
        # return self.observation(self.env.reset(initial_state=initial_state))
        return self.observation(self.env.reset())

class BuildingWrapper(Building):

    def __init__(self, energy_simulation: EnergySimulation, weather: Weather, observation_metadata: Mapping[str, bool], action_metadata: Mapping[str, bool], carbon_intensity: CarbonIntensity = None, pricing: Pricing = None, dhw_storage: StorageTank = None, cooling_storage: StorageTank = None, heating_storage: StorageTank = None, electrical_storage: Battery = None, dhw_device: Union[HeatPump, ElectricHeater] = None, cooling_device: HeatPump = None, heating_device: Union[HeatPump, ElectricHeater] = None, pv: PV = None, name: str = None, **kwargs):
        super().__init__(energy_simulation, weather, observation_metadata, action_metadata, carbon_intensity, pricing, dhw_storage, cooling_storage, heating_storage, electrical_storage, dhw_device, cooling_device, heating_device, pv, name, **kwargs)
        
        self.cooling_storage = StorageTankWrapper(self.cooling_storage.capacity, self.cooling_storage.max_output_power,
                                                    self.cooling_storage.max_input_power)
        self.heating_storage = StorageTankWrapper(self.heating_storage.capacity, self.heating_storage.max_output_power,
                                                    self.heating_storage.max_input_power)
        # Maps keys in observations dict to indexes in the observation vector
        self.obs_idxes = {}
        curr_idx = 0
        for k, v in self.observations().items():
            self.obs_idxes[k] = np.arange(curr_idx, curr_idx + len(v))

    def reset(self, initial_state=None):
        r"""Reset `Building` to initial state."""

        if initial_state is None:
            Building.reset(self)
        else:
            # object reset
            super(Environment, self).reset()
            self.cooling_storage.reset(initial_state[self.obs_idxes["cooling_storage_soc"]])
            self.heating_storage.reset(initial_state[self.obs_idxes["heating_storage_soc"]])
            self.dhw_storage.reset()
            self.electrical_storage.reset()
            self.cooling_device.reset()
            self.heating_device.reset()
            self.dhw_device.reset()
            self.pv.reset()

            # variable reset
            self.__cooling_electricity_consumption = []
            self.__heating_electricity_consumption = []
            self.__dhw_electricity_consumption = []
            self.__solar_generation = self.pv.get_generation(self.energy_simulation.solar_generation)*-1
            self.__net_electricity_consumption = []
            self.__net_electricity_consumption_emission = []
            self.__net_electricity_consumption_price = []
            self.update_variables()

class StorageTankWrapper(StorageTank):
    def reset(self, initial_state=None):
        r"""Reset `StorageDevice` to initial state."""

        super(Environment, self).reset()
        if initial_state is None:
            StorageTank.reset(self)
        else:
            self.__soc = [initial_state.item()]
            self.__energy_balance = [0.0]
