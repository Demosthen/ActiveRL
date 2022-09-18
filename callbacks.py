from ast import Call
from typing import Callable, Dict, Tuple, Union
from stable_baselines3.common.callbacks import BaseCallback
# from state_generation import generate_states
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import RolloutWorker

from state_generation import generate_states
from uncertain_ppo_trainer import UncertainPPO
from simple_grid_wrapper import SimpleGridEnvWrapper
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from reward_predictor import RewardPredictor


class ActiveRLCallback(DefaultCallbacks):
    """
    A custom callback that derives from ``DefaultCallbacks``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, num_descent_steps: int=10, batch_size: int=64, use_coop: bool=True, planning_model=None, config={}, use_gpu=False):
        super(ActiveRLCallback, self).__init__()
        self.num_descent_steps = num_descent_steps
        self.batch_size = batch_size
        self.use_coop = use_coop
        self.planning_model = planning_model
        self.config = config
        self.is_evaluating = False
        self.cell_index = -1
        self.num_cells = -1
        self.is_gridworld = self.config["env"] == SimpleGridEnvWrapper
        self.eval_rewards = []
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y,%H-%M-%S")
        self.use_gpu = use_gpu
        if self.planning_model is not None:
            device = torch.device("cuda:0") if self.use_gpu else torch.device("cpu")
            self.reward_model = RewardPredictor(self.planning_model.obs_size, self.planning_model.hidden_size, self.planning_model.batch_norm, device=device)
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
        def access_eval_metrics(worker):
            if hasattr(worker, "callbacks"):
                worker.callbacks.is_evaluating = False
                return worker.callbacks.eval_rewards
            else:
                return []
        rewards = np.array(algorithm.evaluation_workers.foreach_worker(access_eval_metrics))
        rewards = np.mean(rewards, axis=0)
        evaluation_metrics["evaluation"] = {f"per_cell_rewards{cell}": rew for cell, rew in enumerate(rewards)}
        

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
        # print("DOES THIS RUNNNN\n\n\n\n")
        for env in envs:
            if self.is_evaluating and self.is_gridworld:
                if self.num_cells < 0:
                    self.num_cells = env.ncol * env.nrow
                    self.eval_rewards = [0 for _ in range(self.num_cells)]
                self.cell_index += 1
                env.reset(initial_state=self.cell_index % self.num_cells)

            else:
                new_states, uncertainties = generate_states(policy, obs_space=env.observation_space, num_descent_steps=self.num_descent_steps, 
                batch_size=self.batch_size, use_coop=self.use_coop, planning_model=self.planning_model, reward_model=self.reward_model)
                # TODO: log uncertainties
                new_states = new_states.detach().cpu().flatten()

                # print(env.observation_space)
                env.reset(initial_state=new_states)

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
        envs = base_env.get_sub_environments()
        if self.is_evaluating and self.is_gridworld:
            self.eval_rewards[self.cell_index % self.num_cells] += episode.total_reward

    def on_learn_on_batch(self, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs):
        if self.reward_model is not None:
            obs = torch.tensor(train_batch[SampleBatch.OBS], device=self.reward_model.device)
            rew = torch.tensor(train_batch[SampleBatch.REWARDS], device=self.reward_model.device)
            self.reward_optim.zero_grad()
            rew_hat = self.reward_model(obs).squeeze()
            loss = F.mse_loss(rew, rew_hat)
            # TODO: log this thing
            loss.backward()
            self.reward_optim.step()