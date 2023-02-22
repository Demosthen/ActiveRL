from typing import Dict, Optional, Union
from matplotlib import pyplot as plt
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from ray.rllib.policy import Policy
from ray.rllib.evaluation import RolloutWorker
from core.uncertain_ppo.uncertain_ppo_trainer import UncertainPPO
import numpy as np
from core.constants import *
from core.callbacks import ActiveRLCallback

class SimpleGridCallback(ActiveRLCallback):
    """ Note, SimpleGridCallback is not yet vectorized, so logging may be inaccurate if num_envs_per_worker is not 1"""
    def __init__(self, 
                 num_descent_steps: int = 10, 
                 batch_size: int = 64, 
                 no_coop: bool = False, 
                 planning_model=None, config={}, 
                 run_active_rl=False, 
                 planning_uncertainty_weight=1, 
                 device="cpu", 
                 args={}, 
                 uniform_reset=False):
        super().__init__(num_descent_steps, batch_size, no_coop, planning_model, config, run_active_rl, planning_uncertainty_weight, device, args, uniform_reset)
        self.cell_index = -1
        self.num_cells = -1
        self.eval_rewards = []
        self.goal_reached = []
        self.visualization_env = self.config["env"](self.config["env_config"])
        self.visualization_env.reset()

    def on_evaluate_end(self, *, algorithm: UncertainPPO, evaluation_metrics: dict, **kwargs)-> None:
        """
        Runs at the end of Algorithm.evaluate().
        """
        def access_eval_metrics(worker):
            if hasattr(worker, "callbacks"):
                worker.callbacks.is_evaluating = False
                return worker.callbacks.eval_rewards, worker.callbacks.num_cells
            else:
                return []
        workers_out = algorithm.evaluation_workers.foreach_worker(access_eval_metrics)
        _, self.num_cells = workers_out[0]
        rewards = np.array([output[0] for output in workers_out])
        rewards = np.mean(rewards, axis=0)
        per_cell_rewards = {f"{cell}": rew for cell, rew in enumerate(rewards)}
        evaluation_metrics["evaluation"]["per_cell_rewards"] = per_cell_rewards
        per_cell_rewards["max"] = 10
        per_cell_rewards["min"] = -5
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
        env = base_env.get_sub_environments()[0]
        # Get the single "default policy"
        policy = next(policies.values())
        needs_initialization = self.num_cells < 0
        
        if needs_initialization:
            self.num_cells = env.ncol * env.nrow
            self.eval_rewards = [0 for _ in range(self.num_cells)]
        if self.is_evaluating:
            self.cell_index += 1
            initial_state = np.zeros([self.num_cells])
            initial_state[self.cell_index % self.num_cells] = 1
            env.reset(initial_state=initial_state)
        run_active_rl = np.random.random() < self.run_active_rl
        if not self.is_evaluating and (run_active_rl or self.uniform_reset):
            # Actually run Active RL and reset the environment
            new_states = self.reset_env(policy, env, episode)
            
            if ACTIVE_STATE_VISITATION_KEY not in episode.custom_metrics:
                episode.hist_data[ACTIVE_STATE_VISITATION_KEY] = []
            to_log, _ = env.separate_resettable_part(new_states)
            to_log = np.array(to_log)
            to_log = to_log.argmax()    
            episode.hist_data[ACTIVE_STATE_VISITATION_KEY] = [to_log]
            #episode.hist_data[ACTIVE_STATE_VISITATION_KEY].append(to_log)
            # Limit hist data to last 100 entries so wandb can handle it
            #episode.hist_data[ACTIVE_STATE_VISITATION_KEY] = episode.hist_data[ACTIVE_STATE_VISITATION_KEY][-100:]

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
        if self.is_evaluating:
            self.eval_rewards[self.cell_index % self.num_cells] += episode.total_reward / (self.config["evaluation_duration"] // self.num_cells) 