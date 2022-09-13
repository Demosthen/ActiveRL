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
from model_utils import get_unit
from state_generation import generate_states
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RewardPredictor(nn.Module):
    def __init__(self, in_size, hidden_size, batch_norm: bool=True) -> None:
        super().__init__()
        self.X_mean = nn.Parameter(torch.zeros(in_size), requires_grad=False)
        self.X_std = nn.Parameter(torch.ones(in_size), requires_grad=False)
        self.batch_norm = batch_norm
        self.y_mean = nn.Parameter(torch.zeros([1]), requires_grad=False)
        self.y_std = nn.Parameter(torch.ones([1]), requires_grad=False)
        self.momentum = 0.9
        self.layers = nn.ModuleList([
            get_unit(in_size, hidden_size, batch_norm),
            get_unit(hidden_size, hidden_size, batch_norm),
            get_unit(hidden_size, hidden_size, batch_norm),
            get_unit(hidden_size, hidden_size, batch_norm),
            nn.Linear(hidden_size, 1)
        ])

    def preprocess(self, x):
        ret = (x - self.X_mean.to(self.device)) / self.X_std.to(self.device)
        # Do not update on single samples
        if self.training and len(x) > 1:
            self.X_mean = self.momentum * self.X_mean + (1 - self.momentum) * torch.mean(x)
            self.X_std = self.momentum * self.X_std + (1 - self.momentum) * torch.std(x)
        return ret

    def postprocess(self, y):
        ret = y * self.y_std.to(self.device) + self.y_mean.to(self.device)
        # Do not update on single samples
        if self.training and len(y) > 1:
            self.y_mean = self.momentum * self.y_mean + (1 - self.momentum) * torch.mean(y)
            self.y_std = self.momentum * self.y_std + (1 - self.momentum) * torch.std(y)
        return ret
    
    def forward(self):
        x = self.preprocess(x)
        for i, layer in enumerate(self.layers):
            base = 0
            # Add residual connection if this is not
            # the first or last layer
            if i != 0 and i != len(self.layers) - 1:
                base = x
            x = layer(x) + base
        return self.postprocess(x)

    def eval_batchnorm(self):
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                layer.eval()
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, nn.BatchNorm1d):
                        sublayer.eval()

    def compute_uncertainty(self, in_tensor, num_dropout_evals=10):
        orig_mode = self.training
        self.train()
        self.eval_batchnorm()
        rewards = []
        for _ in range(num_dropout_evals):
            rew = self.forward(in_tensor)
            rewards.append(rew)
        rewards = torch.stack(rewards)
        uncertainty = torch.var(rewards)
        self.train(orig_mode)
        return uncertainty

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
            self.reward_optim = torch.optim.Adam(self.reward_model.parameters(), device=device)
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
            obs = torch.tensor(train_batch[SampleBatch.OBS], self.reward_model.device)
            rew = torch.tensor(train_batch[SampleBatch.REWARDS], self.reward_model.device)
            self.reward_optim.zero_grad()
            rew_hat = self.reward_model(obs)
            loss = F.mse_loss(rew, rew_hat)
            # TODO: log this thing
            loss.backward()
            self.reward_optim.step()