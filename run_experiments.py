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
import pickle
from typing import Callable
import torch
import torch.nn as nn
import argparse
import ray
from citylearn_wrappers.citylearn_wrapper import CityLearnEnvWrapper
from citylearn_wrappers.citylearn_model_training.planning_model import get_planning_model
from core.config import ExperimentConfig
from core.uncertain_ppo.uncertain_ppo_trainer import UncertainPPO
import wandb
# from state_generation import generate_states
from ray.rllib.algorithms.ppo import DEFAULT_CONFIG
from ray.rllib.models.catalog import MODEL_DEFAULTS
from gridworld.simple_grid_wrapper import SimpleGridEnvWrapper
from process_args import add_args, define_constants, process_combo_args
from sinergym_wrappers.sinergym_wrapper import SinergymWrapper

from datetime import datetime
import numpy as np
import random
from dm_maze_wrappers.model import ComplexInputNetwork
from core.utils import *
from core.constants import *
import cProfile
from sinergym_wrappers.epw_scraper.epw_data import EPW_Data


def get_agent(config: ExperimentConfig, args, planning_model=None):

    env = config.env_fn
    callback_fn = config.callback_fn
    rllib_config = config.rllib_config
    env_config = config.env_config
    eval_env_config = config.eval_env_config
    model_config = config.model_config

    config = DEFAULT_CONFIG.copy()
    config["seed"] = args.seed
    config["framework"] = "torch"
    config["env"] = env
    # Disable default preprocessors, we preprocess ourselves with env wrappers
    config["_disable_preprocessor_api"] = True

    config["rollout_fragment_length"] = "auto"
    config["env_config"] = env_config

    config["model"] = MODEL_DEFAULTS
    # Update model config with environment specific config params
    config["model"].update(model_config)
    # Update model config with Active RL related config params
    config["model"]["fcnet_activation"] = lambda: nn.Sequential(
        nn.Tanh(), nn.Dropout(p=args.dropout))  # Custom_Activation
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

    config["callbacks"] = lambda: callback_fn(num_descent_steps=args.num_descent_steps, batch_size=1, no_coop=args.no_coop, planning_model=planning_model,
                                              config=config, run_active_rl=args.use_activerl, planning_uncertainty_weight=args.planning_uncertainty_weight, args=args, uniform_reset=args.use_random_reset)
    config.update(rllib_config)
    agent = UncertainPPO(
        config=config, logger_creator=custom_logger_creator(args.log_path))

    return agent


def train_agent(agent: UncertainPPO, timesteps: int, log_path: str, full_eval_interval: int, config: ExperimentConfig, profile: bool=None, use_wandb=False):
    full_eval_fn = config.full_eval_fn
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
        if i % 1 == 0:
            ckpt_dir = f"{log_path}/checkpoint_{i}"
            path = agent.save(ckpt_dir)
            print(f"SAVING CHECKPOINT TO {path}")
            if use_wandb:
                model_artifact = wandb.Artifact("sinergym_model", type="model")
                model_artifact.add_dir(ckpt_dir)
                wandb.log_artifact(model_artifact)
        if profile is not None:
            print_profile(profile, None)
        training_steps = result["timesteps_total"]

if __name__ == "__main__":
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
        # patching the logdir directly seems to work
        wandb.tensorboard.patch(root_logdir=args.log_path)
        wandb.config.update(args)

    ray.init()

    if args.env == "cl":
        from citylearn_wrappers.config import get_config
    elif args.env == "gw":
        from gridworld.config import get_config
    elif args.env == "dm":
        from dm_maze_wrappers.config import get_config
    elif args.env == "sg":
        from sinergym_wrappers.config import get_config
    else:
        raise NotImplementedError

    config = get_config(args)

    # planning model is None if the ckpt file path is None
    planning_model = get_planning_model(args.planning_model_ckpt)

    agent = get_agent(config, args, planning_model)

    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
    else:
        profile = None

    train_agent(agent, timesteps=args.num_timesteps, log_path=args.log_path,
                full_eval_interval=args.full_eval_interval, config=config, profile=profile, use_wandb=args.wandb)

    if args.profile:
        profile.disable()
        profile_log_path = os.path.join(args.log_path, "profile_log.txt")
        print_profile(profile, profile_log_path)
        wandb.save(profile_log_path)

    result_dict = agent.evaluate()

    if args.env == "gw":
        rewards = result_dict["evaluation"]["per_cell_rewards"]
        print(result_dict["evaluation"]["per_cell_rewards"])
        visualization_env = config.env_fn(config.env_config)
        visualization_env.reset()
        img_arr = visualization_env.render(
            mode="rgb_array", reward_dict=rewards)
        if args.wandb:
            img = wandb.Image(
                img_arr, caption="Rewards from starting from each cell")
            wandb.log({"per_cell_reward_image": img})