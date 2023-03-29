from typing import Dict, Optional, Union
# from state_generation import generate_states
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from ray.rllib.policy import Policy
from ray.rllib.evaluation import RolloutWorker
import numpy as np
from core.constants import *
from ..core.callbacks import ActiveRLCallback

class CitylearnCallback(ActiveRLCallback):
    """ Note, CitylearnCallback is not yet vectorized, so logging may be inaccurate if num_envs_per_worker is not 1"""
    def __init__(self, num_descent_steps: int = 10, batch_size: int = 64, no_coop: bool = False, planning_model=None, config={}, run_active_rl=0, planning_uncertainty_weight=1, device="cpu", args={}, uniform_reset=0):
        super().__init__(num_descent_steps, batch_size, no_coop, planning_model, config, run_active_rl, planning_uncertainty_weight, device, args, uniform_reset)
        
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
    
        if self.is_evaluating:
            #Rotate in the next climate zone
            env.next_env()
        run_active_rl = np.random.random() < self.run_active_rl
        if not self.is_evaluating and (run_active_rl or self.uniform_reset):
            self.reset_env(policy, env, episode)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
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
        env = base_env.get_sub_environments()[0]
        prefix = "eval_" if self.is_evaluating else "train_"
        episode.custom_metrics[prefix + CL_ENV_KEYS[env.curr_env_idx] + "_reward"] = episode.total_reward