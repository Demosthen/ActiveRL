from bisect import insort
from collections import namedtuple
from typing import Dict, Optional, Union
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import RolloutWorker

from core.state_generation import generate_states
from core.uncertain_ppo.uncertain_ppo_trainer import UncertainPPO
import torch
import torch.nn.functional as F
import numpy as np
from core.reward_predictor import RewardPredictor
from core.constants import *
from core.utils import states_to_np
import heapq

class ActiveRLCallback(DefaultCallbacks):
    """
    A custom callback that derives from ``DefaultCallbacks``. Not yet vectorized.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, 
                 num_descent_steps: int=10, 
                 batch_size: int=64, 
                 no_coop: bool=False, 
                 planning_model=None, 
                 config={}, 
                 run_active_rl=0, 
                 planning_uncertainty_weight=1, 
                 device="cpu", 
                 args={}, 
                 uniform_reset = False):
        super().__init__()
        self.run_active_rl = run_active_rl
        self.num_descent_steps = num_descent_steps
        self.batch_size = batch_size
        self.no_coop = no_coop
        self.planning_model = planning_model
        self.config = config
        self.is_evaluating = False
        self.planning_uncertainty_weight = planning_uncertainty_weight
        self.use_gpu = args.num_gpus > 0
        self.args = args
        self.uniform_reset = uniform_reset
        self.full_eval_mode = False
        self.activerl_lr = args.activerl_lr
        self.activerl_reg_coeff = args.activerl_reg_coeff
        self.eval_worker_ids = []
        if self.planning_model is not None:
            device = "cuda:0" if self.use_gpu else "cpu"
            self.reward_model = RewardPredictor(self.planning_model.obs_size, self.config["model"]["fcnet_hiddens"][0], False, device=device)
            self.reward_optim = torch.optim.Adam(self.reward_model.parameters())
        else:
            self.reward_model = None
            self.reward_optim = None

        self.plr_d = args.plr_d
        self.plr_beta = args.plr_beta
        self.env_buffer = []
        if self.plr_d > 0:
            self.plr_scheduler = DecayScheduler(0.5, 0.9)
        self.last_reset_state = None

    def on_evaluate_start(self, *, algorithm: UncertainPPO, **kwargs)-> None:
        """
        This method gets called at the beginning of Algorithm.evaluate().
        """

        def activate_eval_metrics(worker):
            if hasattr(worker, "callbacks"):
                worker.callbacks.is_evaluating = True
            return worker.worker_index
        
        def set_eval_worker_ids(worker):
            if hasattr(worker, "callbacks"):
                worker.callbacks.eval_worker_ids = self.eval_worker_ids

        self.eval_worker_ids = sorted(algorithm.evaluation_workers.foreach_worker(activate_eval_metrics))
        algorithm.evaluation_workers.foreach_worker(set_eval_worker_ids)
        

    def on_evaluate_end(self, *, algorithm: UncertainPPO, evaluation_metrics: dict, **kwargs)-> None:
        """
        Runs at the end of Algorithm.evaluate().
        """
        #self.is_evaluating = False
        def access_eval_metrics(worker):
            if hasattr(worker, "callbacks"):
                worker.callbacks.is_evaluating = False
            else:
                return []
        return algorithm.evaluation_workers.foreach_worker(access_eval_metrics)
        
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
        env = base_env.get_sub_environments()[0]
        # Get the single "default policy"
        policy = next(policies.values())
        run_active_rl = np.random.random() < self.run_active_rl
        if not self.is_evaluating and (run_active_rl or self.uniform_reset):
            self.reset_env(policy, env, episode)

    def reset_env(self, policy, env, episode):
        new_states, uncertainties = generate_states(
            policy, 
            env=env, 
            obs_space=env.observation_space, 
            num_descent_steps=self.num_descent_steps, 
            batch_size=self.batch_size, 
            no_coop=self.no_coop, 
            planning_model=self.planning_model, 
            reward_model=self.reward_model, 
            planning_uncertainty_weight=self.planning_uncertainty_weight, 
            uniform_reset=self.uniform_reset,
            lr=self.activerl_lr,
            plr_d=self.plr_d,
            plr_beta=self.plr_beta,
            env_buffer=self.env_buffer,
            reg_coeff = self.activerl_reg_coeff)
        new_states = states_to_np(new_states)
        episode.custom_metrics[UNCERTAINTY_LOSS_KEY] = uncertainties[-1]
        env.reset(initial_state=new_states)
        self.last_reset_state = new_states
        return new_states

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

    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        result: dict,
        **kwargs,
    ) -> None:
        """Called at the end of Algorithm.train().

        Args:
            algorithm: Current Algorithm instance.
            result: Dict of results returned from Algorithm.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        if self.plr_d > 0:
            vf_loss = result["info"]["learner"]["default_policy"]["learner_stats"]["vf_loss"]
            entry = EnvBufferEntry(np.abs(vf_loss), self.last_reset_state, 1)
            insort(self.env_buffer, entry, key=lambda x: x.value_error)
            # heapq.heappush(self.env_buffer, (np.abs(vf_loss), self.last_reset_state))

    def full_eval(self, algorithm):
        """
            Sets callback into full evaluation mode. Similar to pytorch\'s eval function,
            this does not *actually* run any evaluations
        """
        self.full_eval_mode = True
        def set_full_eval(worker):
            if hasattr(worker, "callbacks"):
                worker.callbacks.full_eval_mode = True
        algorithm.evaluation_workers.foreach_worker(set_full_eval)

    def limited_eval(self, algorithm):
        """
            Sets callback into limited evaluation mode. Similar to pytorch\'s eval function,
            this does not *actually* run any evaluations
        """
        self.full_eval_mode = False
        def set_limited_eval(worker):
            if hasattr(worker, "callbacks"):
                worker.callbacks.full_eval_mode = False
        algorithm.evaluation_workers.foreach_worker(set_limited_eval)
        
class DecayScheduler():
    """Parameter scheduler that anneals the parameter from 0 exponentially to a constant value, then keeps it there."""
    def __init__(self, p, gamma=0.9) -> None:
        self.p = p
        self.gamma = gamma
        self.current_value = 0

    def step(self):
        self.current_value = (self.current_value - self.p) * self.gamma + self.p

class EnvBufferEntry:
    def __init__(self, value_loss, env_params, cnt=1) -> None:
        self.value_error = np.abs(value_loss)
        self.env_params  = env_params
        self.cnt = cnt