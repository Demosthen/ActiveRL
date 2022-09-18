from copy import deepcopy
import os
from enum import Enum
from pathlib import Path
from unittest import result
import gym 
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym_simplegrid.envs.simple_grid import SimpleGridEnvRLLib 
import matplotlib.pyplot as plt
import argparse
import ray
from citylearn.citylearn import CityLearnEnv
from callbacks import ActiveRLCallback
from citylearn_wrapper import CityLearnEnvWrapper
from uncertain_ppo import UncertainPPOTorchPolicy
from uncertain_ppo_trainer import UncertainPPO
from ray.air.callbacks.wandb import WandbLoggerCallback
import wandb
import utils
# from state_generation import generate_states
from ray.rllib.algorithms.ppo import DEFAULT_CONFIG
from ray.rllib.models.catalog import MODEL_DEFAULTS
from ray.rllib.algorithms.callbacks import MultiCallbacks
from simple_grid_wrapper import SimpleGridEnvWrapper
from citylearn_model_training.planning_model import get_planning_model
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_checker import check_env
# from callbacks import ActiveRLCallback
from datetime import datetime

class Environments(Enum):
    GRIDWORLD = "gw"
    CITYLEARN = "cl"


"""SimpleGrid is a super simple gridworld environment for OpenAI gym. It is easy to use and 
customise and it is intended to offer an environment for quick testing and prototyping 
different RL algorithms.

It is also efficient, lightweight and has few dependencies (gym, numpy, matplotlib).

SimpleGrid involves navigating a grid from Start(S) (red tile) to Goal(G) (green tile) 
without colliding with any Wall(W) (black tiles) by walking over the Empty(E) (white tiles) 
cells. The yellow circle denotes the agent's current position.

Optionally, it is possible to introduce a noise in the environment that makes the agent move 
in a random direction that can be different than the desired one.
"""


def read_gridworld(filename):
    with open(filename, 'r') as f:
        grid_str = f.read()
        grid, rew, wind_p = grid_str.split("---")
        grid_desc = eval(grid)
        rew_map = eval(rew)
        wind_p = eval(wind_p)
    return grid_desc, rew_map, wind_p

def initialize_citylearn_params():
    # Load environment
    climate_zone = 5
    buildings = ["Building_1"]

    params = {'root_directory': Path("data/Climate_Zone_" + str(climate_zone)),
            'building_attributes': 'building_attributes.json', 
            'weather_file': 'weather.csv', 
            'solar_profile': 'solar_generation_1kW.csv', 
            'carbon_intensity': 'carbon_intensity.csv',
            'building_ids': buildings,
            'buildings_states_actions': 'buildings_state_action_space.json', 
            'simulation_period': (0, 8760*1-1),
            'cost_function': ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','carbon_emissions'], 
            'central_agent': True,
            'save_memory': False }
    return params


def get_agent(env, env_config, args, planning_model=None):

    config = DEFAULT_CONFIG.copy()
    config["framework"] = "torch"
    config["env"] = env
    # Disable default preprocessors, we preprocess ourselves with env wrappers
    config["_disable_preprocessor_api"] = True
    config["env_config"] = env_config
    if args.env == "cl":
        config["horizon"] = 8760
    config["model"] = MODEL_DEFAULTS
    config["model"]["fcnet_activation"] = lambda: nn.Sequential(nn.Tanh(), nn.Dropout())#Custom_Activation
    config["model"]["num_dropout_evals"] = 10
    config["disable_env_checking"] = True
    config["num_gpus"] = args.num_gpus
    if args.num_gpus == 0:
        config["num_gpus_per_worker"] = 0
    config["evaluation_interval"] = 1
    config["evaluation_num_workers"] = 0
    config["evaluation_duration"] = max(1, args.gw_steps_per_cell) * 64 # TODO: is there a better way of counting this?
    config["evaluation_duration_unit"] = "episodes"
    eval_env_config = deepcopy(env_config)
    eval_env_config["is_evaluation"] = True
    config["evaluation_config"] = {
        "env_config": eval_env_config
    }

    

    # TODO: add callbacks

    if args.use_activerl:
        config["callbacks"] = lambda: ActiveRLCallback(num_descent_steps=args.num_descent_steps, batch_size=1, use_coop=args.use_coop, planning_model=planning_model, config=config)
    agent = UncertainPPO(config = config, logger_creator = utils.custom_logger_creator(args.log_path))

    return agent

def train_agent(agent, timesteps, env):
    training_steps = 0
    while training_steps < timesteps:
        result = agent.train()
        training_steps = result["timesteps_total"]

def get_log_path(log_dir):
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y,%H-%M-%S")
    
    path = os.path.join(".", log_dir, date_time)
    os.makedirs(path, exist_ok=True)
    return path

def add_args(parser):
    # GENERAL PARAMS
    parser.add_argument(
        "--num_gpus",
        type=int,
        help="number of gpus to use, default = 1",
        default=1
        )

    # LOGGING PARAMS
    parser.add_argument(
        "--log_path",
        type=str,
        help="filename to read gridworld specs from. pass an int if you want to auto generate one.",
        default="logs"
        )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="whether or not to log with wandb"
    )
    
    # GENERAL ENV PARAMS
    parser.add_argument(
        "--env",
        type=str,
        help="Which environment: CityLearn, GridWorld, or... future env?",
        default=Environments.GRIDWORLD.value,
        choices=[env.value for env in Environments]
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        help="Number of timesteps to collect from environment during training",
        default=5000
    )

    # GRIDWORLD ENV PARAMS
    parser.add_argument(
        "--gw_filename",
        type=str,
        help="filename to read gridworld specs from. pass an int if you want to auto generate one.",
        default="gridworlds/sample_grid.txt"
        )
    parser.add_argument(
        "--gw_steps_per_cell",
        type=int,
        help="number of times to evaluate each cell, min=1",
        default=1
        )

    # ACTIVE RL PARAMS
    parser.add_argument(
        "--use_activerl",
        type=int,
        help="set to 1 to use the Active RL callback and 0 to not",
        default=0
    )
    parser.add_argument(
        "--num_descent_steps",
        type=int,
        help="How many steps to do gradient descent on state for Active RL",
        default=10
    )
    parser.add_argument(
        "--use_coop",
        action="store_true",
        help="Whether or not to use the constrained optimizer for Active RL optimization"
    )
    parser.add_argument(
        "--planning_model_ckpt",
        type=str,
        help="File path to planning model checkpoint. Leave as None to not use the planning model",
        default=None
    )

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    if args.wandb:
        run = wandb.init(project="active-rl", entity="social-game-rl")
        args.log_path = get_log_path(args.log_path)
        wandb.tensorboard.patch(root_logdir=args.log_path) # patching the logdir directly seems to work
        wandb.config.update(args)
        
    ray.init()

    if args.env == "cl":
        env = CityLearnEnvWrapper
        env_config = {
            "schema": Path("./data/citylearn_challenge_2022_phase_1/schema.json"),
            "planning_model_ckpt": args.planning_model_ckpt,
            "is_evaluation": False
        }
    else: 
        env = SimpleGridEnvWrapper
        env_config = {"is_evaluation": False}
        if args.gw_filename.strip().isnumeric():
            env_config["desc"] = int(args.gw_filename)
        else:
            grid_desc, rew_map, wind_p = read_gridworld(args.gw_filename)
            env_config["desc"] = grid_desc
            env_config["reward_map"] = rew_map
            env_config["wind_p"] = wind_p
        

    # planning model is None if the ckpt file path is None
    planning_model = get_planning_model(args.planning_model_ckpt)

    agent = get_agent(env, env_config, args, planning_model)
    
    train_agent(agent, timesteps=args.num_timesteps, env=env)

    result_dict = agent.evaluate()

    print(result_dict["evaluation"]["custom_metrics"].keys())

    # obs = env.reset()
    # for i in range(16):
    #     action, _state = agent.predict(obs)
    #     pic = env.render(mode="ansi")
    #     print(pic)
    #     #action = int(input("- 0: LEFT - 1: DOWN - 2: RIGHT - 3: UP"))
    #     obs, r, done, info = env.step(action)
    #     if done:
    #         obs = env.reset()
    # plt.imsave("test.png", pic)
    