import _thread
import gym
import numpy as np
from gym.spaces.box import Box
import torch
from core.resettable_env import ResettableEnv
import sinergym
import os
from sinergym_wrappers.sinergym_reward import FangerReward
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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from gym import register
import datetime

LOG_LEVEL_MAIN = 'INFO'
LOG_LEVEL_EPLS = 'FATAL'
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s"

WEATHER_VAR_MAP = {
    'Site Outdoor Air Drybulb Temperature(Environment)': "drybulb",
    'Site Outdoor Air Relative Humidity(Environment)': "relhum",
    'Site Wind Direction(Environment)': "winddir",
    'Site Direct Solar Radiation Rate per Area(Environment)': "dirnorrad",
    'Site Diffuse Solar Radiation Rate per Area(Environment)': "difhorrad",
    'Site Wind Speed(Environment)': "windspd"
}
REVERSE_WEATHER_MAP = {v: k for k, v in WEATHER_VAR_MAP.items()}

class SinergymWrapper(gym.core.ObservationWrapper, ResettableEnv):

    def __init__(self, config):
        """
        Important config parameters:
        use_rbc: replaces all actions with those from the rbc
        use_random: replaces all actions with those from the random controller
        weather_variability: list of dicts that represent weather_variabilities to reset the environment to
            The dictionaries should each map a weather variable name (e.g. 'drybulb') to a tuple of OU process parameters.
            You can specify which weather variability to reset with by calling reset with the initial_state parameter
            set to the index of the weather variability dict you want.
            For example, if you set config["weather_variability"] = [{'drybulb': (1.0, 0, 0.001)}, {'drybulb': (2.0, 0, 0.001)}]
            and call reset(initial_state=0), you would reset using (1.0, 0, 0.001)
        sample_environments: whether or not to sample weather variability from a specified csv file for each parameter
            Set to False by default
        environment_variability_file: file to read weather variabilities to sample from if sample_environments is True
            Set to epw_scraper/US_epw_OU_Params.csv by default
        variability_low: Dict of lower bounds for all specified variability parameters (unnecessary unless you're doing ActiveRL)
        variability_high: Dict of upper bounds for all specified variability parameters (unnecessary unless you're doing ActiveRL)
        act_repeat: the number of times to repeat each action
        """
        # Set up environment
        curr_pid = os.getpid()
        self.base_env_name = 'Eplus-5Zone-hot-discrete-stochastic-v1-FlexibleReset'
        self.timesteps_per_hour = config.get("timesteps_per_hour", 1)

        weather_file = config.get("weather_file", "USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw")
        self.environment_variability_file = config.get("environment_variability_file", "sinergym_wrappers/epw_scraper/US_epw_OU_params.csv")
        self.sample_environments = config.get("sample_environments", False)
        if self.sample_environments:
            self.OU_param_df = self._load_OU_params(self.environment_variability_file)
        self.act_repeat = config.get("act_repeat", 1)
        self.random_week = config["config"].random_week

        self.epw_data = config["epw_data"]
        weather_bounds = {name: (self.epw_data.weather_min[name], self.epw_data.weather_max[name]) for name in self.epw_data.weather_min.index}
        
        # Overrides env_name so initializing multiple Sinergym envs will not result in a race condition
        env = gym.make(self.base_env_name, 
                        env_name=self.base_env_name + str(curr_pid), 
                        weather_file=weather_file,
                        reward=FangerReward,
                        reward_kwargs={
                            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
                            'ppd_variable': 'Zone Thermal Comfort Fanger Model PPD(SPACE1-1 PEOPLE 1)',
                            'occupancy_variable': 'Zone People Occupant Count(SPACE1-1)'
                        },
                        act_repeat=self.act_repeat,
                        config_params={'timesteps_per_hour' : self.timesteps_per_hour,
                                        'weather_bounds': weather_bounds,
                                        "random_week": self.random_week})
        
        # Get controller overrides
        self.use_rbc = config.get("use_rbc", False)
        self.use_random = config.get("use_random", False)

        if not self.use_rbc and not self.use_random:
           env =  NormalizeObservation(env, ranges=self._get_ranges(self.base_env_name))
           obs_space_shape = env.observation_space.shape
           env.observation_space = Box(np.zeros(obs_space_shape), np.ones(obs_space_shape))

        # Read current weather variability
        self.weather_variability = config["weather_variability"]
        self.scenario_idx = 0
        env.weather_variability = self.weather_variability[self.scenario_idx]
        super().__init__(env)
        self.env = env

        # Augment observation space with weather variability info
        self._augment_obs_space(env, config)

        # Initialize baselines if specified.
        if self.use_rbc:
            if "5Zone" in self.base_env_name:
                self.replacement_controller = RBC5Zone(self.env)
            else:
                self.replacement_controller = RBCDatacenter(self.env)
        elif self.use_random:
            self.replacement_controller = RandomController(self.env)
        else:
            self.replacement_controller = None

    def _augment_obs_space(self, env, config):
        obs_space = env.observation_space
        obs_space_shape_list = list(obs_space.shape)
        i = obs_space_shape_list[-1]
        self.original_obs_space_shape = obs_space.shape

        # Maps weather variable name to indexes in observation
        # Has entries for both the epw format (e.g. 'drybulb') and the sinergym format
        # (e.g. 'Site Outdoor Air Drybulb Temperature(Environment)')
        self.variability_noise_idxs = {} 
        self.variability_low = []
        self.variability_high = []
        for key, variability in self.env.weather_variability.items():
            self.variability_noise_idxs[key] = list(range(i, i+2))
            if "variability_low" in config:
                self.variability_low.extend(config["variability_low"][key])
            else:
                self.variability_low.extend(list(variability))
            
            if "variability_high" in config:
                self.variability_high.extend(config["variability_high"][key])
            else:
                self.variability_high.extend(list(variability))
            i += 2
        obs_space_shape_list[-1] = i
        self.variability_offset_idxs = {WEATHER_VAR_MAP.get(variable, ""): i for i, variable in enumerate(self.env.variables["observation"])}

        self.variability_low = np.array(self.variability_low)
        self.variability_high = np.array(self.variability_high)
        self.num_extra_variability_dims = len(self.variability_low)
        self.variability_offset = self.variability_low
        self.variability_scale = (self.variability_high - self.variability_low)
        low = list(obs_space.low) + [0. for _ in range(self.num_extra_variability_dims)]
        high = list(obs_space.high) + [1. for _ in range(self.num_extra_variability_dims)]
        self.observation_space = Box(
            low = np.array(low), 
            high = np.array(high),
            shape = obs_space_shape_list,
            dtype=np.float32)
        
        self.last_untransformed_obs = None
        self.active_variables = list(self.env.weather_variability.keys())

    def _load_OU_params(self, file):
        return pd.read_csv(file)

    def _get_ranges(self, env_name: str):
        if "5Zone" in env_name:
            range = RANGES_5ZONE
            # range['Site Outdoor Air Drybulb Temperature(Environment)'] = [-50., 100.]
            # range['Site Wind Speed(Environment)'] = [0., 50.]
            # range['Zone Air Relative Humidity(SPACE1-1)'] = [0., 100.]
            return range

        elif "office" in env_name:
            return RANGES_OFFICE
        elif "warehouse" in env_name:
            return RANGES_WAREHOUSE
        elif "datacenter" in env_name:
            return RANGES_DATACENTER
        else:
            raise NotImplementedError()

    def observation(self, observation: np.ndarray):
        variability = np.zeros(self.num_extra_variability_dims)

        for key, var in self.env.weather_variability.items():
            idxs = [idx - self.original_obs_space_shape[-1] for idx in self.variability_noise_idxs[key]]
            variability[idxs] = np.array(var)[0::2]
        
        variability = (variability - self.variability_offset) / self.variability_scale
        return np.concatenate([observation, variability], axis=-1)

    # def inverse_observation(self, observation: torch.Tensor):
    #     return observation[..., :-self.num_extra_variability_dims]

    def separate_resettable_part(self, obs: torch.Tensor):
        """Separates the observation into the resettable portion and the original. Make sure this operation is differentiable"""
        resettable = []
        for key in self.active_variables:
            offset = torch.unsqueeze(obs[..., self.variability_offset_idxs[key]], dim=-1)
            resettable.append(offset)
        resettable.append(obs[..., -self.num_extra_variability_dims:])
        resettable = torch.concat(resettable, dim=-1)
        return resettable, obs

    def combine_resettable_part(self, obs: np.ndarray, resettable: np.ndarray):
        """Combines an observation that has been split like in separate_resettable_part back together. Make sure this operation is differentiable"""
        # Make sure torch doesn't backprop into non-resettable part
        obs = obs.detach()
        for i, key in enumerate(self.active_variables):
            obs[..., self.variability_offset_idxs[key]] = resettable[..., i]
        obs[..., -self.num_extra_variability_dims:] = resettable[..., len(self.active_variables):]
        return obs

    def resettable_bounds(self) -> np.ndarray: 
        """Get bounds for resettable part of observation space"""
        low = np.array([0 for _ in range(len(self.active_variables) + self.num_extra_variability_dims)])
        high = np.array([1. for _ in range(self.num_extra_variability_dims + len(self.active_variables))])
        return low, high
    
    def _sample_variability(self):
        row = self.OU_param_df.sample(1)
        ret = {}
        for variable in self.variability_noise_idxs.keys():
            OU_param = [0,0,0]
            for j in range(3):
                OU_param[j] = np.array(row[f"{variable}_{j}"]).squeeze().item()
            ret[variable] = OU_param
        return ret
    
    def _get_range(self, var_name):
        """
        Gets the range of var_name, whether it refers to a weather variable in
        the epw format (e.g. 'drybulb') or the sinergym format
        # (e.g. 'Site Outdoor Air Drybulb Temperature(Environment)')
        """
        if var_name in self.env.ranges:
            return self.env.ranges[var_name]
        elif var_name in REVERSE_WEATHER_MAP:
            return self.env.ranges[REVERSE_WEATHER_MAP[var_name]]
        else:
            raise ValueError(f"Invalid variable {var_name}")
        
    def reset(self, initial_state: Optional[Union[int, np.ndarray]]=None):
        """
        Resets the environment. Pass a tensor with the same shape as the observation as initial_state
        to reset the weather variability to that state. Pass a non-negative int to specify a scenario_idx in the 
        pre-set weather variabilities in the environment. Pass a negative int to sample weather variabilities
        from a file. Pass nothing to use the default weather variability.
        """
        obs = self.env.reset()
        weather_df = self.env.simulator._config.weather_data.get_weather_series()
        self.weather_means = weather_df.mean(axis=0)
        first_day_weather = weather_df.iloc[0]
        self.last_untransformed_obs = obs
        if isinstance(initial_state, int):
            if initial_state < 0 and self.sample_environments:
                curr_weather_variability = self._sample_variability()
            elif initial_state >= 0 and initial_state < len(self.weather_variability):
                # Set to specified weather variability scenario
                self.scenario_idx = initial_state
                curr_weather_variability = self.weather_variability[self.scenario_idx]
            else:
                raise ValueError("initial state does not specify a valid weather variability.")
            print("PRESET VARIABILITY", curr_weather_variability)
            self.env.simulator.reset(curr_weather_variability)
        elif initial_state is not None:
            # Reset simulator with specified weather variability
            variability = self._get_variability_from_state(initial_state)
            variability_dict = {}
            for var_name, idxs in self.variability_noise_idxs.items():
                idxs = [idx - self.original_obs_space_shape[-1] for idx in idxs]
                variability_params = variability[idxs]
                offset = self._get_offset_from_state(initial_state, var_name, first_day_weather)
                variability_dict[var_name] = (variability_params[0], offset, variability_params[1])#(variability_params[0], offset, variability_params[1])
            print("ACTIVE VARIABILITY", variability_dict)
            self.last_variability = variability_dict
            _, obs, _ = self.env.simulator.reset(variability_dict)
            obs = np.array(obs, dtype=np.float32)

        return self.observation(obs)

    def _get_variability_from_state(self, initial_state):
        variability = initial_state[..., -self.num_extra_variability_dims:]
        variability = variability * self.variability_scale + self.variability_offset
        variability = np.clip(variability, self.variability_low, self.variability_high)
        return variability

    def _get_offset_from_state(self, initial_state, var_name, first_day_weather):
        offset_idx = self.variability_offset_idxs[var_name]
        offset = np.clip(initial_state[..., offset_idx], 0, 1)
        var_range = self._get_range(var_name)
        offset = offset * (var_range[1] - var_range[0]) + var_range[0]
        offset -= first_day_weather[var_name]
        return offset
    

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
        
    def sample_obs(self):
        """Automatically sample an observation to seed state generation"""
        obs = np.zeros(self.observation_space.shape) + 0.5
        for var_name, idxs in self.variability_noise_idxs.items():

            offset_idx = self.variability_noise_idxs[var_name]
            var_range = self.env.ranges[REVERSE_WEATHER_MAP[var_name]]
            obs[offset_idx] = (self.weather_means[var_name] - var_range[0]) / (var_range[1] - var_range[0])
        return obs
    
class FlexibleResetConfig(Config):

    def __init__(self,
            idf_path: str,
            weather_path: str,
            variables: Dict[str, List[str]],
            env_name: str,
            max_ep_store: int,
            action_definition: Optional[Dict[str, Any]],
            extra_config: Dict[str, Any]):
        super().__init__(idf_path, weather_path, variables, env_name, max_ep_store, action_definition, extra_config)
        if self.config['random_week']:
            self.next_date_offset = np.random.randint(0, 358) * 24
        else:
            self.next_date_offset = 0

    """ Custom configuration class that extends apply_weather_variability to more flexibly change more weather aspects"""
    def apply_weather_variability(
            self,
            variation: Optional[Dict] = None,
            date_offset: int = 0) -> str:
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

            if date_offset > 0:
                df_copy = df.copy(deep=True)
                df.iloc[date_offset:] = df.iloc[:-date_offset]
                df.iloc[:date_offset] = df_copy[-date_offset:]

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

                if self.config.get("weather_bounds", None) is not None:
                    lower = self.config["weather_bounds"][column][0]
                    upper = self.config["weather_bounds"][column][1]
                else:
                    lower = -np.inf
                    upper = np.inf

                # Create noise
                for i in range(n - 1):
                    x[i + 1] = x[i] + dt * (-(x[i] - mu) / tau) + \
                        sigma_bis * sqrtdt * np.random.randn()

                # Add noise
                df[column] += x
                df[column] = np.clip(df[column], lower, upper)
                
            # Save new weather data
            weather_data_mod.set_weather_series(df)

            filename = self._weather_path.split('/')[-1]
            filename = filename.split('.epw')[0]
            filename += '_Random_%s_%s_%s.epw' % (
                str(sigma), str(mu), str(tau))
            episode_weather_path = self.episode_path + '/' + filename
            weather_data_mod.to_epw(episode_weather_path)
            return episode_weather_path
        
    def _check_eplus_config(self) -> None:
        """Check Eplus Environment config definition is correct.
        """

        # EXTRA CONFIG
        if self.config is not None:
            for config_key in self.config.keys():
                # Check config parameters values
                # Timesteps
                if config_key == 'timesteps_per_hour':
                    assert self.config[config_key] > 0, 'Extra Config: timestep_per_hour must be a positive int value.'
                # Runperiod
                elif config_key == 'runperiod':
                    assert isinstance(
                        self.config[config_key], tuple) and len(
                        self.config[config_key]) == 6, 'Extra Config: Runperiod specified in extra configuration has an incorrect format (tuple with 6 elements).'
                elif config_key == 'weather_bounds':
                    assert isinstance(self.config[config_key], dict)
                elif config_key == "random_week":
                    assert isinstance(self.config[config_key], bool)
                else:
                    raise KeyError(
                        F'Extra Config: Key name specified in config called [{config_key}] has no support in Sinergym.')
        # ACTION DEFINITION
        if self.action_definition is not None:
            for original_sch_name, new_sch in self.action_definition.items():
                # Check action definition format
                assert isinstance(
                    original_sch_name, str), 'Action definition: Keys must be str.'
                assert isinstance(
                    new_sch, dict), 'Action definition: New scheduler definition must be a dict.'
                assert set(
                    new_sch.keys()) == set(
                    ['name', 'initial_value']), 'Action definition: keys in new scheduler definition must be name and initial_value.'
                assert isinstance(
                    new_sch['name'], str), 'Action definition: Name field in new scheduler must be a str element.'
                # Check action definition component is in schedulers available
                # in building model
                assert original_sch_name.lower() in self.schedulers.keys(
                ), 'Action definition: Object called {} is not an existing component in building model.'.format(original_sch_name)
                # Check new variable is present in action variables
                assert new_sch['name'] in self.variables['action'], 'Action definition: {} external variable should be in action variables.'.format(
                    new_sch['name'])


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
        self._config.adapt_variables_to_cfg_and_idf()
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

        self.next_date_offset = 0

    def reset(
        self, weather_variability: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[float, List[float], bool]:
        """Resets the environment.
        This method does the following:
        1. Makes a new EnergyPlus working directory.
        2. Copies .idf and variables.cfg file to the working directory.
        3. Creates the socket.cfg file in the working directory.
        4. Creates the EnergyPlus subprocess.
        5. Establishes the socket connection with EnergyPlus.
        6. Reads the first sensor data from the EnergyPlus.
        7. Uses a new weather file if passed.
        Args:
            weather_variability (Optional[Tuple[float, float, float]], optional): Tuple with the sigma, mean and tau for OU process. Defaults to None.
        Returns:
            Tuple[float, List[float], bool]: The first element is a value with simulation time elapsed;
            the second element consist on EnergyPlus results in a 1-D list corresponding to the variables in
            variables.cfg and year, month, day and hour in simulation. The last element is a boolean indicating whether the episode terminates.
        """
        # End the last episode if exists
        if self._episode_existed:
            self._end_episode()
            self.logger_main.info(
                'EnergyPlus episode completed successfully. ')
            self._epi_num += 1

        # Update current week
        base = datetime.datetime(1991, 1, 1, 0)
        random_offset = datetime.timedelta(days=self.next_date_offset)
        week_delta = datetime.timedelta(days=7)
        start = base + random_offset
        end = start + week_delta
        self._config.config["runperiod"] = (start.day, start.month, end.year, end.day, end.month, end.year)
        self._config.apply_extra_conf()

        # Create EnergyPlus simulation process
        self.logger_main.info('Creating new EnergyPlus simulation episode...')
        # Creating episode working dir
        eplus_working_dir = self._config.set_episode_working_dir()
        # Getting IDF, WEATHER, VARIABLES and OUTPUT path for current episode
        eplus_working_idf_path = self._config.save_building_model()
        eplus_working_var_path = self._config.save_variables_cfg()
        eplus_working_out_path = (eplus_working_dir + '/' + 'output')
        eplus_working_weather_path = self._config.apply_weather_variability(
            variation=weather_variability, date_offset=self.next_date_offset)

        if self._config.config["random_week"]:
            self.next_date_offset = np.random.randint(0, 358) * 24

        self._create_socket_cfg(self._host,
                                self._port,
                                eplus_working_dir)
        # Create the socket.cfg file in the working dir
        self.logger_main.info('EnergyPlus working directory is in %s'
                              % (eplus_working_dir))
        # Create new random weather file in case variability was specified
        # noise always from original EPW

        # Select new weather if it is passed into the method
        eplus_process = self._create_eplus(
            self._eplus_path,
            eplus_working_weather_path,
            eplus_working_idf_path,
            eplus_working_out_path,
            eplus_working_dir)
        self.logger_main.debug(
            'EnergyPlus process is still running ? %r' %
            self._get_is_subprocess_running(eplus_process))
        self._eplus_process = eplus_process

        # Log EnergyPlus output
        eplus_logger = Logger().getLogger('EPLUS_ENV_%s_%s-EPLUSPROCESS_EPI_%d' %
                                          (self._env_name, self._thread_name, self._epi_num), LOG_LEVEL_EPLS, LOG_FMT)
        _thread.start_new_thread(self._log_subprocess_info,
                                 (eplus_process.stdout,
                                  eplus_logger))
        _thread.start_new_thread(self._log_subprocess_err,
                                 (eplus_process.stderr,
                                  eplus_logger))

        # Establish connection with EnergyPlus
        # Establish connection with client
        conn, addr = self._socket.accept()
        self.logger_main.debug('Got connection from %s at port %d.' % (addr))
        # Start the first data exchange
        rcv_1st = conn.recv(2048).decode(encoding='ISO-8859-1')
        self.logger_main.debug(
            'Got the first message successfully: ' + rcv_1st)
        version, flag, nDb, nIn, nBl, curSimTim, Dblist \
            = self._disassembleMsg(rcv_1st)
        # get time info in simulation
        time_info = get_current_time_info(self._config.building, curSimTim)
        # Add time_info date in the end of the Energyplus observation
        Dblist = time_info + Dblist
        # Remember the message header, useful when send data back to EnergyPlus
        self._eplus_msg_header = [version, flag]
        self._curSimTim = curSimTim
        # Check if episode terminates
        is_terminal = False
        if curSimTim >= self._eplus_one_epi_len:
            is_terminal = True
        # Change some attributes
        self._conn = conn
        self._eplus_working_dir = eplus_working_dir
        self._episode_existed = True
        # Check termination
        if is_terminal:
            self._end_episode()


        return (curSimTim, Dblist, is_terminal)

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
        env_entry_point="sinergym_wrappers.sinergym_wrapper:EPlusFlexibleResetEnv"
        env_kwargs = env_spec.kwargs
        if "weather_variability" in env_kwargs:
            # Give weather variability in new format
            env_kwargs["weather_variability"] = {"drybulb": env_kwargs["weather_variability"]}
        register(
            id=env_id, 
            entry_point=env_entry_point, 
            kwargs=env_kwargs)
        