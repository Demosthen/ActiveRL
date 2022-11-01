import ctypes
import ctypes.util
# This line makes sure the dm_maze uses EGL to render, which supports headless setups like savio
ctypes.CDLL(ctypes.util.find_library('GL'), ctypes.RTLD_GLOBAL)

from copy import copy, deepcopy
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
from dm_maze.dm_maze import DM_Maze_Arena, DM_Maze_Env, DM_Maze_Task
from dm_maze.dm_wrapper import DMEnvWrapper
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
import numpy as np
import random
import labmaze
from dm_control.locomotion.walkers import ant

class Environments(Enum):
    GRIDWORLD = "gw"
    CITYLEARN = "cl"
    DM_MAZE = "dm"


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


def get_agent(env, env_config, eval_env_config, args, planning_model=None):
    dummy_env = env(env_config)
    config = DEFAULT_CONFIG.copy()
    config["seed"] = args.seed
    config["framework"] = "torch"
    config["env"] = env
    # Disable default preprocessors, we preprocess ourselves with env wrappers
    config["_disable_preprocessor_api"] = True
    config["env_config"] = env_config
    if args.env == "cl":
        config["horizon"] = 8760
    config["model"] = MODEL_DEFAULTS
    config["model"]["fcnet_activation"] = lambda: nn.Sequential(nn.Tanh(), nn.Dropout(p=args.dropout))#Custom_Activation
    config["model"]["num_dropout_evals"] = args.num_dropout_evals
    config["model"]["shrink_init"] = args.cl_use_rbc_residual
    config["train_batch_size"] = args.train_batch_size
    #config["num_sgd_iter"] = 
    config["disable_env_checking"] = True
    # divide by 5: 1 driver, 2 workers, 1 evaluation workers
    config["num_gpus"] = args.num_gpus / 4
    config["num_gpus_per_worker"] = args.num_gpus / 4
    config["num_workers"] = 2
    config["soft_horizon"] = args.soft_horizon
    config["clip_param"] = args.clip_param
    config["lr"] = args.lr
    if args.num_gpus == 0:
        config["num_gpus_per_worker"] = 0
    config["evaluation_interval"] = args.eval_interval
    config["evaluation_num_workers"] = 1
    if env is SimpleGridEnvWrapper:
        config["evaluation_duration"] = max(1, args.gw_steps_per_cell) * dummy_env.nrow * dummy_env.ncol # TODO: is there a better way of counting this?
    elif env is CityLearnEnvWrapper:
        config["evaluation_duration"] = len(CL_EVAL_PATHS)

    config["evaluation_duration_unit"] = "episodes"
    
    config["evaluation_config"] = {
        "env_config": eval_env_config
    }

    config["callbacks"] = lambda: ActiveRLCallback(num_descent_steps=args.num_descent_steps, batch_size=1, use_coop=args.use_coop, planning_model=planning_model, config=config, run_active_rl=args.use_activerl, planning_uncertainty_weight=args.planning_uncertainty_weight, args=args)
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

def define_constants(args):
    global CL_FOLDER, CL_EVAL_PATHS
    CL_FOLDER = args.cl_eval_folder#"./data/single_building" if args.single_building_eval else "./data/all_buildings"
    CL_EVAL_PATHS = [os.path.join(CL_FOLDER, "Test_cold_Texas/schema.json"), os.path.join(CL_FOLDER, "Test_dry_Cali/schema.json"), os.path.join(CL_FOLDER, "Test_hot_new_york/schema.json"), os.path.join(CL_FOLDER, "Test_snowy_Cali_winter/schema.json")]

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
        help="Which environment: CityLearn, GridWorld, Deepmind Maze or... future env?",
        default=Environments.GRIDWORLD.value,
        choices=[env.value for env in Environments]
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        help="Number of timesteps to collect from environment during training",
        default=5000
    )

    # ALGORITHM PARAMS
    parser.add_argument(
        "--train_batch_size",
        type=int,
        help="Size of training batch",
        default=256
    )
    parser.add_argument(
        "--soft_horizon",
        type=int,
        help="Horizon of timesteps to compute reward over",
        default=48
    )
    parser.add_argument(
        "--clip_param",
        type=float,
        help="PPO clipping parameter",
        default=0.3
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate for PPO",
        default=5e-5
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        help="Number of training steps between evaluation runs",
        default=1
    )
    # CITYLEARN ENV PARAMS
    parser.add_argument(
        "--cl_filename",
        type=str,
        help="schema file to read citylearn specs from.",
        default="./data/citylearn_challenge_2022_phase_1/schema.json"
    )
    parser.add_argument(
        "--cl_eval_folder",
        type=str,
        default="./data/all_buildings",
        help="Which folder\'s building files to evaluate with",
    )
    parser.add_argument(
        "--cl_use_rbc_residual",
        type=int,
        default=0,
        help="Whether or not to train actions as residuals on the rbc"
    )
    parser.add_argument(
        "--cl_action_multiplier",
        type=float,
        default=1,
        help="A scalar to scale the agent's outputs by"
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
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to pass in to rllib workers",
        default=12345678
    )
    parser.add_argument(
        "--planning_uncertainty_weight",
        type=float,
        help="What relative weight to give to the planning uncertainty compared to agent uncertainty",
        default=1
    )
    parser.add_argument(
        "--num_dropout_evals",
        type=int,
        help="Number of dropout evaluations to run to estimate uncertainty",
        default=5
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="Dropout parameter",
        default = 0.5
    )

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    define_constants(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.wandb:
        run = wandb.init(project="active-rl", entity="social-game-rl")
        args.log_path = get_log_path(args.log_path)
        wandb.tensorboard.patch(root_logdir=args.log_path) # patching the logdir directly seems to work
        wandb.config.update(args)
        
    ray.init()

    if args.env == "cl":
        env = CityLearnEnvWrapper
        env_config = {
            "schema": Path(args.cl_filename),
            "planning_model_ckpt": args.planning_model_ckpt,
            "is_evaluation": False,
            "use_rbc_residual": args.cl_use_rbc_residual,
            "action_multiplier": args.cl_action_multiplier
        }
        eval_env_config = deepcopy(env_config)
        eval_env_config["schema"] = [Path(filename) for filename in CL_EVAL_PATHS]
        eval_env_config["is_evaluation"] = True
    elif args.env == "gw": 
        env = SimpleGridEnvWrapper
        env_config = {"is_evaluation": False}
        if args.gw_filename.strip().isnumeric():
            env_config["desc"] = int(args.gw_filename)
        else:
            grid_desc, rew_map, wind_p = read_gridworld(args.gw_filename)
            env_config["desc"] = grid_desc
            env_config["reward_map"] = rew_map
            env_config["wind_p"] = wind_p
        eval_env_config = deepcopy(env_config)
        eval_env_config["is_evaluation"] = True
    elif args.env == "dm":
        env = DMEnvWrapper
        walker = ant.Ant()
        maze_str = "**********\n*.....G..*\n*....P...*\n**********\n"
        arena = DM_Maze_Arena(maze=labmaze.FixedMazeWithRandomGoals(maze_str))
        print(arena._spawn_positions)
        task = DM_Maze_Task(walker, None, arena, 1, enable_global_task_observables=True)
        env_config = {
            "dm_env": DM_Maze_Env,
            "task": task,
            "random_state": np.random.RandomState(42),
            "strip_singleton_obs_buffer_dim": True,
            "time_limit": 10
            }
        #We do not deepcopy here because Mujoco complains 'Cannot copy from elements with attachments'
        eval_env_config = env_config
    else:
        raise NotImplementedError

    # planning model is None if the ckpt file path is None
    planning_model = get_planning_model(args.planning_model_ckpt)

    agent = get_agent(env, env_config, eval_env_config, args, planning_model)
    
    train_agent(agent, timesteps=args.num_timesteps, env=env)

    result_dict = agent.evaluate()

    if args.env == "gw":
        rewards = result_dict["evaluation"]["per_cell_rewards"]
        print(result_dict["evaluation"]["per_cell_rewards"])
        visualization_env = env(env_config)
        visualization_env.reset()
        img_arr = visualization_env.render(mode="rgb_array", reward_dict=rewards)
        if args.wandb:
            img = wandb.Image(img_arr, caption="Rewards from starting from each cell")
            wandb.log({"per_cell_reward_image": img})

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
    