import gym
import numpy as np
from gym.spaces.box import Box
from resettable_env import ResettableEnv
import sinergym
import os
from sinergym_reward import FangerReward
from sinergym.utils.controllers import RBC5Zone, RBCDatacenter, RandomController
from sinergym.utils.wrappers import NormalizeObservation
from sinergym.utils.constants import *
from sinergym.utils.rewards import *
from sinergym.utils.common import *
from sinergym.utils.config import Config
from sinergym.utils.logger import Logger
from sinergym.simulators.eplus import EnergyPlus
from sinergym.envs.eplus_env import EplusEnv
from sinergym.utils.controllers import RBC5Zone
from copy import deepcopy
import socket
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
from gym import register


LOG_LEVEL_MAIN = 'INFO'
LOG_LEVEL_EPLS = 'FATAL'
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s"
class SinergymWrapper(gym.core.ObservationWrapper, ResettableEnv):

    def __init__(self, config):
        curr_pid = os.getpid()
        self.base_env_name = 'Eplus-5Zone-hot-discrete-stochastic-v1-FlexibleReset'
        # Overrides env_name so initializing multiple Sinergym envs will not result in a race condition
        
        env = gym.make(self.base_env_name, 
                        env_name=self.base_env_name + str(os.getpid()), 
                        reward=FangerReward,
                        reward_kwargs={
                            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
                            'ppd_variable': 'Zone Thermal Comfort Fanger Model PPD(SPACE1-1 PEOPLE 1)',
                            'occupancy_variable': 'Zone People Occupant Count(SPACE1-1)'
                        })
        if not config["use_rbc"] and not config["use_random"]:
           env =  NormalizeObservation(env, ranges=self._get_ranges(self.base_env_name))
        self.weather_variability = config["weather_variability"]
        self.scenario_idx = 0
        env.weather_variability = self.weather_variability[self.scenario_idx]
        super().__init__(env)
        self.env = env
        # Augment observation space with weather variability info
        obs_space = env.observation_space
        obs_space_shape_list = list(obs_space.shape)
        obs_space_shape_list[-1] += 3
        self.variability_low = np.array([0., -25, 0.00099])
        self.variability_high = np.array([20., 25, 0.00011])
        self.variability_offset = (self.variability_high + self.variability_low) / 2
        self.variability_scale = (self.variability_high - self.variability_low) / 2
        low = list(obs_space.low) + [-1., -1., -1.]#self.variability_low
        high = list(obs_space.high) + [1., 1., 1.]#self.variability_high
        self.observation_space = Box(
            low = np.array(low), 
            high = np.array(high),
            shape = obs_space_shape_list,
            dtype=np.float32)
        self.is_evaluation = config["is_evaluation"]
        self.last_untransformed_obs = None
        if config["use_rbc"]:
            if "5Zone" in self.base_env_name:
                self.replacement_controller = RBC5Zone(self.env)
            else:
                self.replacement_controller = RBCDatacenter(self.env)
        elif config["use_random"]:
            self.replacement_controller = RandomController(self.env)
        else:
            self.replacement_controller = None

    def _get_ranges(self, env_name):
        if "5Zone" in env_name:
            return RANGES_5ZONE
        elif "pffice" in env_name:
            return RANGES_OFFICE
        elif "warehouse" in env_name:
            return RANGES_WAREHOUSE
        elif "datacenter" in env_name:
            return RANGES_DATACENTER
        else:
            raise NotImplementedError()

    def observation(self, observation):
        #TODO: make compatible with variability being a dict, and flexible as to what to vary
        variability = np.array(self.env.weather_variability)
        variability = (variability - self.variability_offset) / self.variability_scale
        return np.concatenate([observation, variability], axis=-1)

    def inverse_observation(self, observation):
        return observation[..., :-3]

    def separate_resettable_part(self, obs):
        """Separates the observation into the resettable portion and the original. Make sure this operation is differentiable"""
        if obs is None:
            return self.env.weather_variability
        return obs[..., -3:-1], obs

    def combine_resettable_part(self, obs, resettable):
        """Combines an observation that has been split like in separate_resettable_part back together. Make sure this operation is differentiable"""
        # Make sure torch doesn't backprop into non-resettable part
        obs = obs.detach()
        obs[..., -3:-1] = resettable
        return obs

    def resettable_bounds(self):
        """Get bounds for resettable part of observation space"""
        low = np.array([-1., -1.])#, 0.])
        high = np.array([1., 1.])#., 5.])
        return low, high

    def reset(self, initial_state=None):
        obs = self.env.reset()
        self.last_untransformed_obs = obs
        if initial_state is not None:
            # Reset simulator with specified weather variability
            variability = initial_state[..., -3:]
            variability = variability * self.variability_scale + self.variability_offset
            variability[..., -1] = 0.001
            print("ACTIVE VARIABILITY", variability)
            #TODO: make variability a dict
            _, obs, _ = self.env.simulator.reset(tuple(variability))
            obs = np.array(obs, dtype=np.float32)
        else:
            # Cycle through the weather variability scenarios
            curr_weather_variability = self.weather_variability[self.scenario_idx]
            print("PRESET VARIABILITY", curr_weather_variability)
            #TODO: make variability a dict
            self.env.simulator.reset(curr_weather_variability)
            self.scenario_idx = (self.scenario_idx + 1) % len(self.weather_variability)
        return self.observation(obs)

    def step(self, action):
        """Returns modified observations and inputs modified actions"""
        action = self.replace_action(self.last_untransformed_obs, action)
        obs, reward, done, info = self.env.step(action)
        self.last_untransformed_obs = obs
        return self.observation(obs), reward, done, info

    def replace_action(self, obs, action):
        """Replace RL Controller\'s actions with those from a baseline controller"""
        if self.replacement_controller is None:
            return action
        elif isinstance(self.replacement_controller, RandomController):
            return self.replacement_controller.act()
        else:
            return self.replacement_controller.act(obs)

class FlexibleResetConfig(Config):
    """ Custom configuration class that extends apply_weather_variability to more flexibly change more weather aspects"""
    def apply_weather_variability(
            self,
            variation: Optional[Dict] = None) -> str:
        """Modify weather data using Ornstein-Uhlenbeck process.
        Args:
            variation (Dict, optional): Maps columns to be affected to the corresponding Ornstein-Uhlenbeck process parameters.
                The OU parameters should be specified as a Tuple with the sigma, mean and tau for the OU process.
                For example, one could pass {'drybulb': (1, 0, 0.001)} to make the drybulb temperatures change according to an OU with
                parameters 1, 0, and 0.001 and sigma, mean, and tau respectively.
        Returns:
            str: New EPW file path generated in simulator working path in that episode or current EPW path if variation is not defined.
        """
        if variation is None:
            return self._weather_path
        else:
            # deepcopy for weather_data
            weather_data_mod = deepcopy(self.weather_data)
            # Get dataframe with weather series
            df = weather_data_mod.get_weather_series()
            for column, col_variation in variation.items():
                sigma = col_variation[0]  # Standard deviation.
                mu = col_variation[1]  # Mean.
                tau = col_variation[2]  # Time constant.

                T = 1.  # Total time.
                # All the columns are going to have the same num of rows since they are
                # in the same dataframe
                n = len(df[column])
                dt = T / n
                # t = np.linspace(0., T, n)  # Vector of times.

                sigma_bis = sigma * np.sqrt(2. / tau)
                sqrtdt = np.sqrt(dt)

                x = np.zeros(n)

                # Create noise
                for i in range(n - 1):
                    x[i + 1] = x[i] + dt * (-(x[i] - mu) / tau) + \
                        sigma_bis * sqrtdt * np.random.randn()

                # Add noise
                df[column] += x

            # Save new weather data
            weather_data_mod.set_weather_series(df)

            filename = self._weather_path.split('/')[-1]
            filename = filename.split('.epw')[0]
            filename += '_Random_%s_%s_%s.epw' % (
                str(sigma), str(mu), str(tau))
            episode_weather_path = self.episode_path + '/' + filename
            weather_data_mod.to_epw(episode_weather_path)
            return episode_weather_path


class EPlusFlexibleResetSimulator(EnergyPlus):
    """Exactly the same as EplusEnv, except the reset function can reset things other than drybulb temps"""
    def __init__(
            self,
            eplus_path: str,
            weather_path: str,
            bcvtb_path: str,
            idf_path: str,
            env_name: str,
            variables: Dict[str, List[str]],
            act_repeat: int = 1,
            max_ep_data_store_num: int = 10,
            action_definition: Optional[Dict[str, Any]] = None,
            config_params: Optional[Dict[str, Any]] = None):
        """EnergyPlus simulation class.
        Args:
            eplus_path (str):  EnergyPlus installation path.
            weather_path (str): EnergyPlus weather file (.epw) path.
            bcvtb_path (str): BCVTB installation path.
            idf_path (str): EnergyPlus input description file (.idf) path.
            env_name (str): The environment name.
            variables (Dict[str,List[str]]): Variables list with observation and action keys in a dictionary.
            act_repeat (int, optional): The number of times to repeat the control action. Defaults to 1.
            max_ep_data_store_num (int, optional): The number of simulation results to keep. Defaults to 10.
            config_params (Optional[Dict[str, Any]], optional): Dictionary with all extra configuration for simulator. Defaults to None.
        """
        self._env_name = env_name
        self._thread_name = threading.current_thread().getName()
        self.logger_main = Logger().getLogger(
            'EPLUS_ENV_%s_%s_ROOT' %
            (env_name, self._thread_name), LOG_LEVEL_MAIN, LOG_FMT)

        # Set the environment variable for bcvtb
        os.environ['BCVTB_HOME'] = bcvtb_path
        # Create a socket for communication with the EnergyPlus
        self.logger_main.debug('Creating socket for communication...')
        self._socket = socket.socket()
        # Get local machine name
        self._host = socket.gethostname()
        # Bind to the host and any available port
        self._socket.bind((self._host, 0))
        # Get the port number
        sockname = self._socket.getsockname()
        self._port = sockname[1]
        # Listen on request
        self._socket.listen(60)

        self.logger_main.debug(
            'Socket is listening on host %s port %d' % (sockname))

        # Path attributes
        self._eplus_path = eplus_path
        self._weather_path = weather_path
        self._idf_path = idf_path
        # Episode existed
        self._episode_existed = False

        self._epi_num = 0
        self._act_repeat = act_repeat
        self._max_ep_data_store_num = max_ep_data_store_num
        self._last_action = [21.0, 25.0]

        # Creating models config (with extra params if exits)
        self._config = FlexibleResetConfig(
            idf_path=self._idf_path,
            weather_path=self._weather_path,
            variables=variables,
            env_name=self._env_name,
            max_ep_store=self._max_ep_data_store_num,
            action_definition=action_definition,
            extra_config=config_params)

        # Annotate experiment path in simulator
        self._env_working_dir_parent = self._config.experiment_path
        # Setting an external interface if IDF building has not got.
        self.logger_main.info(
            'Updating idf ExternalInterface object if it is not present...')
        self._config.set_external_interface()
        # Updating IDF file (Location and DesignDays) with EPW file
        self.logger_main.info(
            'Updating idf Site:Location and SizingPeriod:DesignDay(s) to weather and ddy file...')
        self._config.adapt_idf_to_epw()
        # Updating IDF file Output:Variables with observation variables
        # specified in environment and variables.cfg construction
        self.logger_main.info(
            'Updating idf OutPut:Variable and variables XML tree model for BVCTB connection.')
        self._config. adapt_variables_to_cfg_and_idf()
        # Setting up extra configuration if exists
        self.logger_main.info(
            'Setting up extra configuration in building model if exists...')
        self._config.apply_extra_conf()
        # Setting up action definition automatic manipulation if exists
        self.logger_main.info(
            'Setting up action definition in building model if exists...')
        self._config.adapt_idf_to_action_definition()

        # In this lines Epm model is modified but no IDF is stored anywhere yet

        # Eplus run info
        (self._eplus_run_st_mon,
         self._eplus_run_st_day,
         self._eplus_run_st_year,
         self._eplus_run_ed_mon,
         self._eplus_run_ed_day,
         self._eplus_run_ed_year,
         self._eplus_run_st_weekday,
         self._eplus_n_steps_per_hour) = self._config._get_eplus_run_info()

        # Eplus one epi len
        self._eplus_one_epi_len = self._config._get_one_epi_len()
        # Stepsize in seconds
        self._eplus_run_stepsize = 3600 / self._eplus_n_steps_per_hour

class EPlusFlexibleResetEnv(EplusEnv):
    """Exactly the same as EplusEnv, except the reset function can reset things other than drybulb temps"""
    def __init__(
        self,
        idf_file: str,
        weather_file: str,
        observation_space: gym.spaces.Box = gym.spaces.Box(
            low=-5e6, high=5e6, shape=(4,)),
        observation_variables: List[str] = [],
        action_space: Union[gym.spaces.Box, gym.spaces.Discrete] = gym.spaces.Box(
            low=0, high=0, shape=(0,)),
        action_variables: List[str] = [],
        action_mapping: Dict[int, Tuple[float, ...]] = {},
        weather_variability: Optional[Dict] = None,
        reward: Any = LinearReward,
        reward_kwargs: Optional[Dict[str, Any]] = {},
        act_repeat: int = 1,
        max_ep_data_store_num: int = 10,
        action_definition: Optional[Dict[str, Any]] = None,
        env_name: str = 'eplus-env-v1',
        config_params: Optional[Dict[str, Any]] = None
    ):
        """Environment with EnergyPlus simulator.
        Args:
            idf_file (str): Name of the IDF file with the building definition.
            weather_file (str): Name of the EPW file for weather conditions.
            observation_space (gym.spaces.Box, optional): Gym Observation Space definition. Defaults to an empty observation_space (no control).
            observation_variables (List[str], optional): List with variables names in IDF. Defaults to an empty observation variables (no control).
            action_space (Union[gym.spaces.Box, gym.spaces.Discrete], optional): Gym Action Space definition. Defaults to an empty action_space (no control).
            action_variables (List[str],optional): Action variables to be controlled in IDF, if that actions names have not been configured manually in IDF, you should configure or use extra_config. Default to empty List.
            action_mapping (Dict[int, Tuple[float, ...]], optional): Action mapping list for discrete actions spaces only. Defaults to empty list.
            weather_variability (Optional[Tuple[float]], optional): Tuple with sigma, mu and tao of the Ornstein-Uhlenbeck process to be applied to weather data. Defaults to None.
            reward (Any, optional): Reward function instance used for agent feedback. Defaults to LinearReward.
            reward_kwargs (Optional[Dict[str, Any]], optional): Parameters to be passed to the reward function. Defaults to empty dict.
            act_repeat (int, optional): Number of timesteps that an action is repeated in the simulator, regardless of the actions it receives during that repetition interval.
            max_ep_data_store_num (int, optional): Number of last sub-folders (one for each episode) generated during execution on the simulation.
            action_definition (Optional[Dict[str, Any]): Dict with building components to being controlled by Sinergym automatically if it is supported. Default value to None.
            env_name (str, optional): Env name used for working directory generation. Defaults to eplus-env-v1.
            config_params (Optional[Dict[str, Any]], optional): Dictionary with all extra configuration for simulator. Defaults to None.
        """

        # ---------------------------------------------------------------------------- #
        #                          Energyplus, BCVTB and paths                         #
        # ---------------------------------------------------------------------------- #
        eplus_path = os.environ['EPLUS_PATH']
        bcvtb_path = os.environ['BCVTB_PATH']
        self.pkg_data_path = PKG_DATA_PATH

        self.idf_path = os.path.join(self.pkg_data_path, 'buildings', idf_file)
        self.weather_path = os.path.join(
            self.pkg_data_path, 'weather', weather_file)

        # ---------------------------------------------------------------------------- #
        #                             Variables definition                             #
        # ---------------------------------------------------------------------------- #
        self.variables = {}
        self.variables['observation'] = observation_variables
        self.variables['action'] = action_variables

        # ---------------------------------------------------------------------------- #
        #                                   Simulator                                  #
        # ---------------------------------------------------------------------------- #
        self.simulator = EPlusFlexibleResetSimulator(
            env_name=env_name,
            eplus_path=eplus_path,
            bcvtb_path=bcvtb_path,
            idf_path=self.idf_path,
            weather_path=self.weather_path,
            variables=self.variables,
            act_repeat=act_repeat,
            max_ep_data_store_num=max_ep_data_store_num,
            action_definition=action_definition,
            config_params=config_params
        )

        # ---------------------------------------------------------------------------- #
        #                       Detection of controllable planners                     #
        # ---------------------------------------------------------------------------- #
        self.schedulers = self.get_schedulers()

        # ---------------------------------------------------------------------------- #
        #        Adding simulation date to observation (not needed in simulator)       #
        # ---------------------------------------------------------------------------- #

        self.variables['observation'] = ['year', 'month',
                                         'day', 'hour'] + self.variables['observation']

        # ---------------------------------------------------------------------------- #
        #                              Weather variability                             #
        # ---------------------------------------------------------------------------- #
        self.weather_variability = weather_variability

        # ---------------------------------------------------------------------------- #
        #                               Observation Space                              #
        # ---------------------------------------------------------------------------- #
        self.observation_space = observation_space

        # ---------------------------------------------------------------------------- #
        #                                 Action Space                                 #
        # ---------------------------------------------------------------------------- #
        # Action space type
        self.flag_discrete = (
            isinstance(
                action_space,
                gym.spaces.Discrete))

        # Discrete
        if self.flag_discrete:
            self.action_mapping = action_mapping
            self.action_space = action_space
        # Continuous
        else:
            # Defining action values setpoints (one per value)
            self.setpoints_space = action_space

            self.action_space = gym.spaces.Box(
                # continuous_action_def[2] --> shape
                low=np.repeat(-1, action_space.shape[0]),
                high=np.repeat(1, action_space.shape[0]),
                dtype=action_space.dtype
            )

        # ---------------------------------------------------------------------------- #
        #                                    Reward                                    #
        # ---------------------------------------------------------------------------- #
        self.reward_fn = reward(self, **reward_kwargs)
        self.obs_dict = None

        # ---------------------------------------------------------------------------- #
        #                        Environment definition checker                        #
        # ---------------------------------------------------------------------------- #

        self._check_eplus_env()

# Add new variants of all registered sinergym envs
registered_envs = list(gym.envs.registry.items())
for id, env_spec in registered_envs:
    if "Eplus" in id and "FlexibleReset" not in id:
        env_id = env_spec.id + "-FlexibleReset"
        env_entry_point="sinergym_wrapper:EPlusFlexibleResetEnv"
        env_kwargs = env_spec.kwargs
        if "weather_variability" in env_kwargs:
            # Give weather variability in new format
            env_kwargs["weather_variability"] = {"drybulb": env_kwargs["weather_variability"]}
        register(
            id=env_id, 
            entry_point=env_entry_point, 
            kwargs=env_kwargs)