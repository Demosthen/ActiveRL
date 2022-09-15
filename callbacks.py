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
from uncertain_ppo_trainer import UncertainPPO
from simple_grid_wrapper import SimpleGridEnvWrapper
from datetime import datetime

class ActiveRLCallback(DefaultCallbacks):
    """
    A custom callback that derives from ``DefaultCallbacks``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, num_descent_steps: int=10, batch_size: int=64, use_coop: bool=True, planning_model=None, config={}):
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
        
        print("IS THIS CALLBACK EVEN INITIALIZED", date_time)


    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """

    def on_evaluate_start(self, *, algorithm: UncertainPPO, **kwargs)-> None:
        """
        This method gets called at the beginning of Algorithm.evaluate().
        """
        self.is_evaluating = True
        print("DOES EVALUATION EVEN STARTT", self.is_evaluating)
        if self.num_cells > 0:
            self.eval_rewards = [0 for _ in range(self.num_cells)]

    def on_evaluate_end(self, *, algorithm: UncertainPPO, evaluation_metrics: dict, **kwargs)-> None:
        """
        Runs at the end of Algorithm.evaluate().
        """
        self.is_evaluating = False
        print("EVAL REWWWWWWWAAAAAAAAARRRRRRRRRRRDDDDDDDDDDSSSSSSSS\n\n\n", self.eval_rewards)
        evaluation_metrics["per_cell_rewards"] = self.eval_rewards

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
            if self.is_evaluating:
                print("does this run\n\n\n\n")
            if self.is_evaluating and self.is_gridworld:
                print("eval episode \n\n\n\n")
                if self.num_cells < 0:
                    self.num_cells = env.ncol * env.nrow
                    print(self.num_cells, env.ncol, env.nrow)
                    self.eval_rewards = [0 for _ in range(self.num_cells)]
                self.cell_index += 1
                env.reset(initial_state=self.cell_index % self.num_cells)

            else:
                new_states, uncertainties = generate_states(policy, obs_space=env.observation_space, num_descent_steps=self.num_descent_steps, 
                batch_size=self.batch_size, use_coop=self.use_coop, planning_model=self.planning_model)
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
        if self.is_evaluating and self.is_gridworld:
            self.eval_rewards[self.cell_index % self.num_cells] += episode.total_reward


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