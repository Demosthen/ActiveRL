from copy import deepcopy

import wandb
from sinergym_wrappers.epw_scraper.epw_data import EPW_Data
from core.utils import get_variability_configs
from sinergym_wrappers.callbacks import SinergymCallback
from sinergym_wrappers.sinergym_wrapper import SinergymWrapper
from core.config import ExperimentConfig
import math
from core.constants import SG_WEATHER_TYPES
def get_config(args):
    rllib_config = {}
    model_config = {}
    eval_env_config = {}
    
    if args.only_drybulb:
        weather_var_names = ['drybulb']
        weather_var_rev_names = []
    else:
        weather_var_names = ['drybulb', 'relhum',
                                "winddir", "dirnorrad"]
        weather_var_rev_names = ["windspd"]

    epw_data = EPW_Data.load("sinergym_wrappers/epw_scraper/US_epw_OU_data.pkl")
    # We only need to include the default evaluation variability since we'll sample the rest later
    weather_var_config = get_variability_configs(
        weather_var_names, weather_var_rev_names, only_default_eval=args.sample_envs, epw_data=epw_data, no_noise=args.no_noise)
    
    base_weather_file = select_weather_file(args.base_weather)
    env_config = {
        # sigma, mean, tau for OU Process
        "weather_variability": weather_var_config["train_var"],
        "variability_low": weather_var_config["train_var_low"],
        "variability_high": weather_var_config["train_var_high"],
        "use_rbc": args.use_rbc,
        "use_random": args.use_random,
        "sample_environments": args.sample_envs,
        "timesteps_per_hour": args.sinergym_timesteps_per_hour,
        "weather_file": base_weather_file,
        "epw_data": epw_data,
        "continuous": args.continuous,
        "random_month": args.random_month,
        "only_vary_offset": args.only_vary_offset,
    }
    eval_env_config = deepcopy(env_config)
    eval_env_config["weather_variability"] = weather_var_config["eval_var"]
    eval_env_config["timesteps_per_hour"] = args.sinergym_timesteps_per_hour * args.eval_fidelity_ratio
    eval_env_config["act_repeat"] = args.eval_fidelity_ratio

    if args.wandb:
        wandb.config.update({
            "train_weather_variability": env_config["weather_variability"],
            "eval_weather_variability": eval_env_config["weather_variability"],
            "weather_variables": weather_var_names + weather_var_rev_names})

    # rllib_config["evaluation_duration"] = len(
    #     eval_env_config["weather_variability"]) * args.horizon
    if args.sample_envs:
        rllib_config["evaluation_duration"] = args.num_eval_workers * args.num_envs_per_worker * args.horizon * 2
    else:
        rllib_config["evaluation_duration"] = args.num_eval_workers * args.num_envs_per_worker * args.horizon * 2 * math.ceil(len(weather_var_config["eval_var"]) / args.num_envs_per_worker)
    rllib_config["evaluation_duration_unit"] = "timesteps"
    rllib_config["horizon"] = args.horizon
    rllib_config["soft_horizon"] = not args.random_month # If we are using random weeks we want a hard horizon to ensure resets
    rllib_config["restart_failed_sub_environments"] = True
    rllib_config["evaluation_sample_timeout_s"] = 3600
    # rllib_config["recreate_failed_workers"] = True
    #rllib_config["batch_mode"] = "complete_episodes"
    rllib_config["evaluation_parallel_to_training"] = True
    return ExperimentConfig(SinergymWrapper, SinergymCallback, 
                            env_config=env_config,
                            rllib_config=rllib_config, 
                            model_config=model_config, 
                            eval_env_config=eval_env_config)

def select_weather_file(weather_choice):
    if weather_choice == SG_WEATHER_TYPES.HOT.value:
        return 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw'
    elif weather_choice == SG_WEATHER_TYPES.MIXED.value:
        return 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw'
    elif weather_choice == SG_WEATHER_TYPES.COOL.value:
        return "USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw"
    else:
        raise NotImplementedError()