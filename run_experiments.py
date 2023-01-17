import ctypes
import ctypes.util
import os
# This line makes sure the dm_maze uses EGL to render, which supports headless setups like savio
# ctypes.CDLL(ctypes.util.find_library('GL'), ctypes.RTLD_GLOBAL)
# os.environ["MUJOCO_GL"] = "egl"
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
from copy import copy, deepcopy

from enum import Enum
from pathlib import Path
import torch
import torch.nn as nn
import argparse
import ray
from citylearn_wrapper import CityLearnEnvWrapper
from citylearn_model_training.planning_model import get_planning_model
from uncertain_ppo_trainer import UncertainPPO
import wandb
import utils
# from state_generation import generate_states
from ray.rllib.algorithms.ppo import DEFAULT_CONFIG
from ray.rllib.models.catalog import MODEL_DEFAULTS
from simple_grid_wrapper import SimpleGridEnvWrapper
from sinergym_wrapper import SinergymWrapper

from datetime import datetime
import numpy as np
import random
from dm_maze.model import ComplexInputNetwork
from utils import flatten_dict_of_lists, read_gridworld, grid_desc_to_dm, print_profile
from constants import *
import cProfile

class Environments(Enum):
    GRIDWORLD = "gw"
    CITYLEARN = "cl"
    DM_MAZE = "dm"
    SINERGYM = "sg"

def get_agent(env, callback_fn, rllib_config, env_config, eval_env_config, model_config, args, planning_model=None):
    
    config = DEFAULT_CONFIG.copy()
    config["seed"] = args.seed
    config["framework"] = "torch"
    config["env"] = env
    # Disable default preprocessors, we preprocess ourselves with env wrappers
    config["_disable_preprocessor_api"] = True
    config["env_config"] = env_config
    

    config["model"] = MODEL_DEFAULTS
    # Update model config with environment specific config params
    config["model"].update(model_config)
    # Update model config with Active RL related config params
    config["model"]["fcnet_activation"] = lambda: nn.Sequential(nn.Tanh(), nn.Dropout(p=args.dropout))#Custom_Activation
    config["model"]["num_dropout_evals"] = args.num_dropout_evals

    config["train_batch_size"] = args.train_batch_size
    config["num_sgd_iter"] = args.num_sgd_iter
    config["disable_env_checking"] = True
    # 1 driver, N training workers, 1 evaluation workers
    total_workers = args.num_training_workers + 1 + args.num_eval_workers
    config["num_gpus"] = args.num_gpus / total_workers
    config["num_gpus_per_worker"] = args.num_gpus / total_workers
    config["num_workers"] = args.num_training_workers
    config["num_envs_per_worker"] = args.num_envs_per_worker
    
    config["clip_param"] = args.clip_param
    config["lr"] = args.lr
    config["gamma"] = args.gamma
    if args.num_gpus == 0:
        config["num_gpus_per_worker"] = 0
    config["evaluation_interval"] = args.eval_interval
    config["evaluation_num_workers"] = args.num_eval_workers
    
    config["evaluation_duration_unit"] = "episodes"
    
    config["evaluation_config"] = {
        "env_config": eval_env_config
    }

    config["callbacks"] = lambda: callback_fn(num_descent_steps=args.num_descent_steps, batch_size=1, no_coop=args.no_coop, planning_model=planning_model, config=config, run_active_rl=args.use_activerl, planning_uncertainty_weight=args.planning_uncertainty_weight, args=args)
    config.update(rllib_config)
    agent = UncertainPPO(config = config, logger_creator = utils.custom_logger_creator(args.log_path))

    return agent

def train_agent(agent, timesteps, full_eval_interval, full_eval_fn = None, profile=None):
    training_steps = 0
    i = 0
    while training_steps < timesteps:
        i += 1
        result = agent.train()
        if i % full_eval_interval == 0 and full_eval_fn is not None:
            agent.callbacks.full_eval(agent)
            result["full_evaluation"] = full_eval_fn(agent)["evaluation"]
            agent._result_logger.on_result(result)
            agent.callbacks.limited_eval(agent)
        if profile is not None:
            print_profile(profile, None)
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

def process_combo_args(args):
    if args.sinergym_sweep is not None:
        arglist = args.sinergym_sweep.split(",")
        if len(arglist) != 3:
            raise ValueError("sinergym sweep combo arg does not have three elements")
        args.use_activerl = float(arglist[0])
        args.use_rbc = int(arglist[1])
        args.use_random = int(arglist[2])
    return args

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
    parser.add_argument(
        "--project",
        type=str,
        help="wandb project to send metrics to",
        default="active-rl"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Whether to profile this run to debug performance"
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
        "--horizon", "--soft_horizon",
        type=int,
        help="Horizon of timesteps to compute reward over (WARNING, I don't think this does anything for dm_control maze)",
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
        "--gamma",
        type=float,
        help="Gamma for PPO",
        default=0.99
    )
    parser.add_argument(
        "--num_sgd_iter",
        type=int,
        help="num_sgd_iter for PPO",
        default=30
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        help="Number of training steps between evaluation runs",
        default=1
    )
    parser.add_argument(
        "--num_training_workers",
        type=int,
        help="Number of workers to collect data during training",
        default=3
    )
    parser.add_argument(
        "--num_eval_workers",
        type=int,
        help="Number of workers to collect data during eval",
        default=1
    )
    parser.add_argument(
        "--num_envs_per_worker",
        type=int,
        help="Number of workers to collect data during training",
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

    # DM MAZE ENV PARAMS
    parser.add_argument(
        "--dm_filename",
        type=str,
        help="filename to read gridworld specs from. pass an int if you want to auto generate one.",
        default="gridworlds/sample_grid.txt"
        )

    parser.add_argument(
        "--aliveness_reward",
        type=float,
        help="Reward to give agent for staying alive",
        default=0.01
        )

    parser.add_argument(
        "--distance_reward_scale",
        type=float,
        help="Scale of reward for agent getting closer to goal",
        default=0.01
        )

    parser.add_argument(
        "--use_all_geoms",
        action="store_true",
        help="Whether to use all possible geoms or try to consolidate them for faster speed (if you turn this on things will slow down significantly)",
        )

    parser.add_argument(
        "--walker",
        type=str,
        choices=["ant", "ball"],
        help="What type of walker to use. Ant and ball are currently supported",
        default="ant"
        )
    parser.add_argument(
        "--dm_steps_per_cell",
        type=int,
        help="number of times to evaluate each cell, min=1",
        default=1
        )
    parser.add_argument(
        "--control_timestep",
        type=float,
        help="Time between control timesteps in seconds",
        default=0.1#DEFAULT_CONTROL_TIMESTEP
        )
    parser.add_argument(
        "--physics_timestep",
        type=float,
        help="Time between physics timesteps in seconds",
        default=0.02#DEFAULT_PHYSICS_TIMESTEP
        )

    # SINERGYM ENV PARAMS
    parser.add_argument(
        "--use_rbc",
        type=int,
        help="Whether or not to override all actions with that of Rule Based Controller (RBC). Set to 1 to enable.",
        default=0
        )
    parser.add_argument(
        "--use_random",
        type=int,
        help="Whether or not to override all actions with that of Random Controller. Set to 1 to enable.",
        default=0
        )

    # ACTIVE RL PARAMS
    parser.add_argument(
        "--use_activerl",
        type=float,
        help="Probability of a training episode using an active start. Set to 1 to only use the active start and 0 to use default",
        default=0
    )
    parser.add_argument(
        "--num_descent_steps",
        type=int,
        help="How many steps to do gradient descent on state for Active RL",
        default=10
    )
    parser.add_argument(
        "--no_coop",
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
    parser.add_argument(
        "--full_eval_interval",
        type=int,
        help="Number of training steps between full evaluation runs (i.e. visiting every state in gridworld or mujoco maze)",
        default=10
    )
    # Combo args for wandb sweeps
    parser.add_argument(
        "--sinergym_sweep",
        type=str,
        help="Sets use_activerl, use_rbc, and use_random, respectively, all at the same time. Pass in the values as a comma delimited string. \
            For example, \'0.1,0,0\' denotes setting use_activerl to 0.1 and use_rbc and use_random to 0",
        default=None
    )

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    args = process_combo_args(args)
    define_constants(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.wandb:
        run = wandb.init(project=args.project, entity="social-game-rl")
        args.log_path = get_log_path(args.log_path)
        wandb.tensorboard.patch(root_logdir=args.log_path) # patching the logdir directly seems to work
        wandb.config.update(args)
        
    ray.init()

    model_config = {}
    rllib_config = {}

    full_eval_fn = None
    if args.env == "cl":
        from callbacks import CitylearnCallback

        callback_fn = CitylearnCallback
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

        rllib_config["horizon"] = args.horizon
        rllib_config["evaluation_duration"] = len(CL_EVAL_PATHS)
        rllib_config["soft_horizon"] = False

    elif args.env == "gw": 
        from callbacks import SimpleGridCallback

        callback_fn = SimpleGridCallback
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

        dummy_env = env(env_config)
        rllib_config["evaluation_duration"] = max(1, args.gw_steps_per_cell) * dummy_env.nrow * dummy_env.ncol # TODO: is there a better way of counting this?

        # model_config["shrink_init"] = args.cl_use_rbc_residual # NOTE: This does not actually do anything anymore
    elif args.env == "dm":
        from callbacks import DMMazeCallback
        from dm_maze.dm_wrapper import DM_Maze_Obs_Wrapper
        callback_fn = DMMazeCallback
        grid_desc, rew_map, wind_p = read_gridworld(args.dm_filename)
        maze_str, subtarget_rews = grid_desc_to_dm(grid_desc, rew_map, wind_p)
        env = DM_Maze_Obs_Wrapper
        env_config = {
            "maze_str": maze_str,
            "subtarget_rews": subtarget_rews,
            "random_state": np.random.RandomState(42),
            "strip_singleton_obs_buffer_dim": True,
            "time_limit": args.horizon * args.control_timestep,
            "aliveness_reward": args.aliveness_reward,
            "distance_reward_scale": args.distance_reward_scale,
            "use_all_geoms": args.use_all_geoms,
            "walker": args.walker,
            "control_timestep": args.control_timestep,
            "physics_timestep": args.physics_timestep
            }

        eval_env_config = deepcopy(env_config)

        dummy_env = env(env_config)

        model_config["dim"] = 32
        model_config["conv_filters"] = [
            [8, [8, 8], 4],
            [16, [3, 3], 2],
            [32, [8, 8], 1],
        ]
        model_config["post_fcnet_hiddens"] = [64, 64]
        model_config["custom_model"] = ComplexInputNetwork

        rllib_config["evaluation_duration"] = 1
        rllib_config["horizon"] = args.horizon
        rllib_config["keep_per_episode_custom_metrics"] = False
        rllib_config["batch_mode"] = "truncate_episodes"
        rllib_config["evaluation_sample_timeout_s"] = 600
        rllib_config["rollout_fragment_length"] = 1000
        dummy_arena = dummy_env.get_task()._maze_arena
        grid_positions = flatten_dict_of_lists(dummy_arena.find_token_grid_positions(RESPAWNABLE_TOKENS))
        rllib_config["evaluation_duration"] = args.dm_steps_per_cell
        rllib_config["evaluation_parallel_to_training"] = True
        full_eval_duration = max(1, args.dm_steps_per_cell) * len(grid_positions)
        full_eval_fn = lambda agent: agent.evaluate(lambda x: full_eval_duration - x)
        # rllib_config["record_env"] = True
    elif args.env == "sg":
        from callbacks import SynergymCallback
        callback_fn = SynergymCallback
        env = SinergymWrapper
        env_config = {
            "is_evaluation": False,
            # sigma, mean, tau for OU Process
            "weather_variability": [(1.0, 0.0, 0.001)],
            "use_rbc": args.use_rbc,
            "use_random": args.use_random
            }

        eval_env_config = deepcopy(env_config)
        eval_env_config["is_evaluation"] = True
        eval_env_config["weather_variability"] = [(1, -30, 0.001),
                                                  (1, 30, 0.001),
                                                  (20, 0, 0.001)]

        rllib_config["evaluation_duration"] = len(eval_env_config["weather_variability"])
        rllib_config["horizon"] = args.horizon
        rllib_config["batch_mode"] = "complete_episodes"
        rllib_config["evaluation_parallel_to_training"] = True
    else:
        raise NotImplementedError

    # planning model is None if the ckpt file path is None
    planning_model = get_planning_model(args.planning_model_ckpt)

    agent = get_agent(env, callback_fn, rllib_config, env_config, eval_env_config, model_config, args, planning_model)
    
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    train_agent(agent, timesteps=args.num_timesteps, full_eval_interval=args.full_eval_interval, full_eval_fn=full_eval_fn, profile=profile)

    if args.profile:
        profile.disable()
        profile_log_path = os.path.join(args.log_path, "profile_log.txt")
        print_profile(profile, profile_log_path)
        wandb.save(profile_log_path)

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
    
