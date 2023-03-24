from typing import Dict, Optional, Union
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from ray.rllib.policy import Policy
from ray.rllib.evaluation import RolloutWorker
import numpy as np
from core.constants import *
from core.callbacks import ActiveRLCallback
class SinergymCallback(ActiveRLCallback):
    def __init__(self, 
                 num_descent_steps: int=10, 
                 batch_size: int=64, 
                 no_coop: bool=False, 
                 planning_model=None, 
                 config={}, 
                 run_active_rl=False, 
                 planning_uncertainty_weight=1, 
                 device="cpu", 
                 args={}, 
                 uniform_reset=False):
        super().__init__(num_descent_steps, batch_size, no_coop, 
                         planning_model, config, run_active_rl, 
                         planning_uncertainty_weight, device, args, 
                         uniform_reset)
        self.num_envs = config["num_envs_per_worker"]
        self.env_to_scenario_index = {k: -1 for k in range(self.num_envs)}
        self.sample_environments = config["env_config"].get("sample_environments", False)
        self.scenario_index = 0

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
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
        episode.user_data["power"] = []
        episode.user_data["term_comfort"] = []
        episode.user_data["term_energy"] = []
        episode.user_data["num_comfort_violations"] = 0
        episode.user_data["out_temperature"] = []
        episode.user_data[f"reward"]= []

        env = base_env.get_sub_environments()[env_index]
        # Get the single "default policy"
        policy = next(policies.values())
        run_active_rl = np.random.random() < self.run_active_rl
        if not self.is_evaluating and any([run_active_rl, self.uniform_reset, self.plr_d > 0]):
            self.reset_env(policy, env, episode)
        elif self.is_evaluating:
            is_default_env_worker = (worker.worker_index == self.eval_worker_ids[0]) and env_index == 0
            if self.sample_environments and not is_default_env_worker:
                # Set scenario_index to -2 to sample weather variability.
                # We also want to make sure the default environment is represented,
                # so let one environment reset with the default variability.
                scenario_index = -2
                # self.scenario_index = (self.scenario_index + 1) % (self.num_envs * len(self.eval_worker_ids))
            else:
                scenario_index = self.scenario_index
                self.scenario_index = (self.scenario_index + 1) % len(env.weather_variability)
            print("WHAT IS MY SCENARIO INDEX???", scenario_index)
            env.reset(scenario_index)
            self.env_to_scenario_index[env_index] = scenario_index
            

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Runs on each episode step.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects.
                In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index: The index of the sub-environment that stepped the episode
                (within the vector of sub-environments of the BaseEnv).
            kwargs: Forward compatibility placeholder.
        """
        env = base_env.get_sub_environments()[env_index]
        obs = episode.last_observation_for()
        info = episode.last_info_for()
        episode.user_data["power"].append(info["total_power"])
        episode.user_data["term_comfort"].append(info["comfort_penalty"])
        episode.user_data["term_energy"].append(info["total_power_no_units"])
        episode.user_data["out_temperature"].append(info["out_temperature"])
        episode.user_data[f"reward"].append(episode.last_reward_for())
        if info["comfort_penalty"] != 0:
            episode.user_data["num_comfort_violations"] += 1

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
        to_log = {}
        to_log["cum_power"] = np.sum(episode.user_data["power"])
        to_log["mean_power"] = np.mean(episode.user_data["power"])
        to_log["cum_comfort_penalty"] = np.sum(episode.user_data["term_comfort"])
        to_log["mean_comfort_penalty"] = np.mean(episode.user_data["term_comfort"])
        to_log["cum_power_penalty"] = np.sum(episode.user_data["term_energy"])
        to_log["mean_power_penalty"] = np.mean(episode.user_data["term_energy"])
        to_log["num_comfort_violations"] = episode.user_data["num_comfort_violations"]
        to_log["out_temperature_mean"] = np.mean(episode.user_data["out_temperature"])
        to_log["out_temperature_std"] = np.std(episode.user_data["out_temperature"])
        to_log["reward_mean"] = np.mean(episode.user_data["reward"])
        to_log["reward_sum"] = np.sum(episode.user_data["reward"])
        episode.hist_data["out_temperature"] = episode.user_data["out_temperature"][::6000]
        
        try:
            to_log['comfort_violation_time(%)'] = episode.user_data["num_comfort_violations"] / \
                episode.length * 100
        except ZeroDivisionError:
            to_log['comfort_violation_time(%)'] = np.nan

        # Log both scenario specific and aggregated logs
        episode.custom_metrics.update(to_log)
        scenario_index = self.env_to_scenario_index[env_index]
        env_specific_log = {f"env_{scenario_index}_{key}": val for key, val in to_log.items()}
        episode.custom_metrics.update(env_specific_log)