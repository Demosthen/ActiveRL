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
    def __init__(self, num_descent_steps: int=10, batch_size: int=64, use_coop: bool=True, planning_model=None, use_gpu=False):
        super(ActiveRLCallback, self).__init__()
        self.num_descent_steps = num_descent_steps
        self.batch_size = batch_size
        self.use_coop = use_coop
        self.planning_model = planning_model
        self.use_gpu = use_gpu
        if self.planning_model is not None:
            device = torch.device("cuda:0") if self.use_gpu else torch.device("cpu")
            self.reward_model = RewardPredictor(self.planning_model.obs_size, self.planning_model.hidden_size, self.planning_model.batch_norm, device=device)
            self.reward_optim = torch.optim.Adam(self.reward_model.parameters())
        else:
            self.reward_model = None
            self.reward_optim = None

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
        # This should be just one env for now since we are working with a central policy
        for env in envs:
            new_states, uncertainties = generate_states(policy, obs_space=env.observation_space, num_descent_steps=self.num_descent_steps, 
            batch_size=self.batch_size, use_coop=self.use_coop, planning_model=self.planning_model, reward_model=self.reward_model)
            # TODO: log uncertainties
            new_states = new_states.detach().cpu().flatten()

            # print(env.observation_space)
            env.reset(initial_state=new_states)

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