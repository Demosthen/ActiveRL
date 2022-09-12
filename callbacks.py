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
from ray.rllib.evaluation import RolloutWorker

from state_generation import generate_states

class ActiveRLCallback(DefaultCallbacks):
    """
    A custom callback that derives from ``DefaultCallbacks``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, num_descent_steps: int=10, batch_size: int=64, use_coop: bool=True, planning_model=None):
        super(ActiveRLCallback, self).__init__()
        self.num_descent_steps = num_descent_steps
        self.batch_size = batch_size
        self.use_coop = use_coop
        self.planning_model = planning_model

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

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
            
            new_states, uncertainties = generate_states(policy, obs_space=env.observation_space, num_descent_steps=self.num_descent_steps, 
            batch_size=self.batch_size, use_coop=self.use_coop, planning_model=self.planning_model)
            print(new_states.shape)
            new_states = new_states.detach().cpu().flatten()

            # print(env.observation_space)
            env.reset(initial_state=new_states)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass