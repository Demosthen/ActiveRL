from ast import Call
from cProfile import run
from tkinter import ACTIVE
from typing import Callable, Dict, Tuple, Union
# from state_generation import generate_states
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import RolloutWorker
from citylearn_wrapper import CityLearnEnvWrapper

from state_generation import generate_states
from uncertain_ppo_trainer import UncertainPPO
from simple_grid_wrapper import SimpleGridEnvWrapper
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from reward_predictor import RewardPredictor
import logging
from PIL import Image
import wandb
from torch.utils.tensorboard import SummaryWriter

ACTIVE_STATE_VISITATION_KEY = "active_state_visitation"
UNCERTAINTY_LOSS_KEY = "uncertainty_loss"
CL_ENV_KEYS = ["cold_Texas", "dry_Cali", "hot_new_york", "snowy_Cali_winter"]

class ActiveRLCallback(DefaultCallbacks):
    """
    A custom callback that derives from ``DefaultCallbacks``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, num_descent_steps: int=10, batch_size: int=64, use_coop: bool=True, planning_model=None, config={}, run_active_rl=False, planning_uncertainty_weight=1, device="cpu", args={}):
        super(ActiveRLCallback, self).__init__()
        self.run_active_rl = run_active_rl
        self.num_descent_steps = num_descent_steps
        self.batch_size = batch_size
        self.use_coop = use_coop
        self.planning_model = planning_model
        self.config = config
        self.is_evaluating = False
        self.cell_index = -1
        self.num_cells = -1
        self.is_gridworld = self.config["env"] == SimpleGridEnvWrapper
        self.is_citylearn = self.config["env"] == CityLearnEnvWrapper
        self.planning_uncertainty_weight = planning_uncertainty_weight
        self.eval_rewards = []
        self.use_gpu = args.num_gpus > 0
        self.args = args
        self.visualization_env = self.config["env"](self.config["env_config"])
        self.visualization_env.reset()
        if self.planning_model is not None:
            device = "cuda:0" if self.use_gpu else "cpu"
            self.reward_model = RewardPredictor(self.planning_model.obs_size, self.config["model"]["fcnet_hiddens"][0], False, device=device)
            self.reward_optim = torch.optim.Adam(self.reward_model.parameters())
        else:
            self.reward_model = None
            self.reward_optim = None

    def on_evaluate_start(self, *, algorithm: UncertainPPO, **kwargs)-> None:
        """
        This method gets called at the beginning of Algorithm.evaluate().
        """
        self.is_evaluating = True
        
        if self.num_cells > 0:
            self.eval_rewards = [0 for _ in range(self.num_cells)]

        def activate_eval_metrics(worker):
            if hasattr(worker, "callbacks"):
                worker.callbacks.is_evaluating = True
                if worker.callbacks.num_cells > 0:
                    worker.callbacks.eval_rewards = [0 for _ in range(worker.callbacks.num_cells)]
        
        algorithm.evaluation_workers.foreach_worker(activate_eval_metrics)

    def on_evaluate_end(self, *, algorithm: UncertainPPO, evaluation_metrics: dict, **kwargs)-> None:
        """
        Runs at the end of Algorithm.evaluate().
        """
        self.is_evaluating = False
        if self.is_gridworld:
            def access_eval_metrics(worker):
                if hasattr(worker, "callbacks"):
                    worker.callbacks.is_evaluating = False
                    return worker.callbacks.eval_rewards
                else:
                    return []
            rewards = np.array(algorithm.evaluation_workers.foreach_worker(access_eval_metrics))
            rewards = np.mean(rewards, axis=0)
            per_cell_rewards = {f"{cell}": rew for cell, rew in enumerate(rewards)}
            evaluation_metrics["evaluation"]["per_cell_rewards"] = per_cell_rewards
            img_arr = self.visualization_env.render(mode="rgb_array", reward_dict=per_cell_rewards)
            img_arr = np.transpose(img_arr, [2, 0, 1])
            evaluation_metrics["evaluation"]["per_cell_rewards_img"] = img_arr[None, None, :, :, :]

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        **kwargs,
    ) -> None:
        """Callback run on the rollout worker before each episode starts.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode: Episode object which contains the episode's
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        envs = base_env.get_sub_environments()
        # Get the single "default policy"
        policy = next(policies.values())
        for env in envs:
            if self.is_evaluating and self.is_gridworld:
                if self.num_cells < 0:
                    self.num_cells = env.ncol * env.nrow
                    self.eval_rewards = [0 for _ in range(self.num_cells)]
                self.cell_index += 1
                initial_state = np.zeros([self.num_cells])
                initial_state[self.cell_index % self.num_cells] = 1
                env.reset(initial_state=initial_state)

            elif not self.is_evaluating and self.run_active_rl:
                new_states, uncertainties = generate_states(policy, obs_space=env.observation_space, num_descent_steps=self.num_descent_steps, 
                            batch_size=self.batch_size, use_coop=self.use_coop, planning_model=self.planning_model, reward_model=self.reward_model, planning_uncertainty_weight=self.planning_uncertainty_weight)
                new_states = new_states.detach().cpu().flatten()
                episode.custom_metrics[UNCERTAINTY_LOSS_KEY] = uncertainties[-1].loss.detach().cpu().numpy()
                # print(env.observation_space)
                env.reset(initial_state=new_states)
                if self.is_gridworld:
                    if ACTIVE_STATE_VISITATION_KEY not in episode.custom_metrics:
                        episode.hist_data[ACTIVE_STATE_VISITATION_KEY] = []
                    episode.hist_data[ACTIVE_STATE_VISITATION_KEY].append(new_states.numpy().argmax())
            elif self.is_evaluating and self.is_citylearn:
                #Rotate in the next climate zone
                env.next_env()
                
            if self.is_citylearn:
                episode.hist_data[CL_ENV_KEYS[env.curr_env_idx]] = []

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        **kwargs)-> None:
        """
        Runs when an episode is done.

        Args:
            worker: Reference to the current rollout worker.
            base_env : BaseEnv running the episode. The underlying sub environment
                objects can be retrieved by calling base_env.get_sub_environments().
            policies : Mapping of policy id to policy objects. In single agent mode
                there will only be a single “default_policy”.
            episode : Episode object which contains episode state. You can use the
                episode.user_data dict to store temporary data, and episode.custom_metrics
                to store custom metrics for the episode. In case of environment failures,
                episode may also be an Exception that gets thrown from the environment
                before the episode finishes. Users of this callback may then handle
                these error cases properly with their custom logics.
            kwargs : Forward compatibility placeholder.
        """
        if self.is_evaluating and self.is_gridworld:
            self.eval_rewards[self.cell_index % self.num_cells] += episode.total_reward / (self.config["evaluation_duration"] // self.num_cells) 
        if self.is_citylearn:
            env = base_env.get_unwrapped()[0]
            episode.custom_metrics[CL_ENV_KEYS[env.curr_env_idx] + "_reward"] = episode.total_reward #/ self.config["evaluation_duration"]


    def on_learn_on_batch(self, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs):
        """
        Runs each time the agent is updated on a batch of data (Note this also applies to PPO's minibatches)

        Args:
            worker: Reference to the current rollout worker.
            policies : Mapping of policy id to policy objects. In single agent mode
                there will only be a single “default_policy”.
            train_batch: Batch of training data
            result : Dictionary to add custom metrics into
            kwargs : Forward compatibility placeholder.
        """
        if self.reward_model is not None:
            obs = torch.tensor(train_batch[SampleBatch.OBS], device=self.reward_model.device)
            rew = torch.tensor(train_batch[SampleBatch.REWARDS], device=self.reward_model.device)
            self.reward_optim.zero_grad()
            rew_hat = self.reward_model(obs).squeeze()
            loss = F.mse_loss(rew, rew_hat)
            result["reward_predictor_loss"] = loss.detach().cpu().numpy()
            loss.backward()
            self.reward_optim.step()
