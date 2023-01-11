from copy import deepcopy
from typing import Callable, Dict, Optional, Tuple, Union
from matplotlib import pyplot as plt
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
from constants import *
from dm_maze.dm_wrapper import DM_Maze_Obs_Wrapper
from utils import states_to_np, flatten_dict_of_lists
from constants import DEFAULT_REW_MAP
from skimage.transform import resize
from viz_utils import draw_box
class ActiveRLCallback(DefaultCallbacks):
    """
    A custom callback that derives from ``DefaultCallbacks``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, num_descent_steps: int=10, batch_size: int=64, no_coop: bool=False, planning_model=None, config={}, run_active_rl=0, planning_uncertainty_weight=1, device="cpu", args={}):
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
        self.limited_eval()
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

        def activate_eval_metrics(worker):
            if hasattr(worker, "callbacks"):
                worker.callbacks.is_evaluating = True
            return None
        algorithm.evaluation_workers.foreach_worker(activate_eval_metrics)
        

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
        if not self.is_evaluating and run_active_rl:
            self.reset_env(policy, env, episode)

    def reset_env(self, policy, env, episode):
        new_states, uncertainties = generate_states(policy, env=env, obs_space=env.observation_space, num_descent_steps=self.num_descent_steps, 
                        batch_size=self.batch_size, no_coop=self.no_coop, planning_model=self.planning_model, reward_model=self.reward_model, planning_uncertainty_weight=self.planning_uncertainty_weight)
        new_states = states_to_np(new_states)
        episode.custom_metrics[UNCERTAINTY_LOSS_KEY] = uncertainties[-1]
        env.reset(initial_state=new_states)
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

    def full_eval(self):
        """
            Sets callback into full evaluation mode. Similar to pytorch\'s eval function,
            this does not *actually* run any evaluations
        """
        self.full_eval_mode = True

    def limited_eval(self):
        """
            Sets callback into limited evaluation mode. Similar to pytorch\'s eval function,
            this does not *actually* run any evaluations
        """
        self.full_eval_mode = False

class SimpleGridCallback(ActiveRLCallback):
    def __init__(self, num_descent_steps: int = 10, batch_size: int = 64, no_coop: bool = False, planning_model=None, config={}, run_active_rl=False, planning_uncertainty_weight=1, device="cpu", args={}):
        super().__init__(num_descent_steps, batch_size, no_coop, planning_model, config, run_active_rl, planning_uncertainty_weight, device, args)
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
        if not self.is_evaluating and run_active_rl:
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

class CitylearnCallback(ActiveRLCallback):
    def __init__(self, num_descent_steps: int = 10, batch_size: int = 64, no_coop: bool = False, planning_model=None, config={}, run_active_rl=False, planning_uncertainty_weight=1, device="cpu", args={}):
        super().__init__(num_descent_steps, batch_size, no_coop, planning_model, config, run_active_rl, planning_uncertainty_weight, device, args)
        
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
        if not self.is_evaluating and run_active_rl:
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

class DMMazeCallback(ActiveRLCallback):

    def __init__(self, num_descent_steps: int = 10, batch_size: int = 64, no_coop: bool = False, planning_model=None, config={}, run_active_rl=False, planning_uncertainty_weight=1, device="cpu", args={}):
        super().__init__(num_descent_steps, batch_size, no_coop, planning_model, config, run_active_rl, planning_uncertainty_weight, device, args)
        self.cell_index = -1
        self.num_cells = -1
        self.eval_rewards = []
        self.goal_reached = []

    def _get_img_bounds(self, img):
        """ Get bounds of foreground of image (assuming background is uniform and starts at edge of image)"""
        bg_value = img.flatten()[0]
        img = deepcopy(img)
        foreground_prev = np.nonzero(img)
        img[np.abs(img - bg_value) < 10] = 0
        foreground = np.nonzero(img)
        x_low, x_high = np.min(foreground[-2]), np.max(foreground[-2]) + 1
        y_low, y_high = np.min(foreground[-1]), np.max(foreground[-1]) + 1
        return x_low, x_high, y_low, y_high

    def _generate_highlighted_img(self, img: np.ndarray, per_cell_rewards: Dict, shift: float, scale: float):
        """Generates an image overlaid with a grid highlighted according to per_cell_rewards.
            WARNING: this is currently in-place."""
        x_low, x_high, y_low, y_high = self._get_img_bounds(img[:, :, :, :, :img.shape[-2]])
        highlight_img_h = x_high - x_low
        highlight_img_w = y_high - y_low
        highlight_img = np.zeros([3, highlight_img_h, highlight_img_w])
        for idx, reward in per_cell_rewards.items():
            grid_pos = self.grid_positions[int(idx)]
            scaled_reward = (reward + shift) * scale#(reward + 10000) / 300
            draw_box(highlight_img, grid_pos, self.grid_h, self.grid_w, scaled_reward)

        img[..., :, x_low:x_high, y_low:y_high] = img[..., :, x_low:x_high, y_low:y_high] * 0.75 + 0.25 * np.uint8(highlight_img)
        return img

    def on_evaluate_end(self, *, algorithm: UncertainPPO, evaluation_metrics: dict, **kwargs)-> None:
        """
        Runs at the end of Algorithm.evaluate().
        """
        def access_eval_metrics(worker):
            if hasattr(worker, "callbacks") and worker.callbacks.num_cells > 0:
                worker.callbacks.is_evaluating = False
                return worker.callbacks.eval_rewards, worker.callbacks.goal_reached, worker.callbacks.grid_h, worker.callbacks.grid_w, worker.callbacks.grid_positions, worker.callbacks.world_positions, worker.callbacks.num_cells
            else:
                return []
        workers_out = [out for out in algorithm.evaluation_workers.foreach_worker(access_eval_metrics) if len(out) > 0]
        _, _, self.grid_h, self.grid_w, self.grid_positions, \
            self.world_positions, self.num_cells = workers_out[0]
        rewards = np.array([output[0] for output in workers_out])
        goal_reached = np.array([output[1] for output in workers_out])
        goal_reached_vid = evaluation_metrics["evaluation"]["episode_media"]["img"][0]
        goal_reached_img = goal_reached_vid[-1][None, None, :, :, :]
        evaluation_metrics["evaluation"]["single_img"] = goal_reached_img
        evaluation_metrics["evaluation"]["vid"] = goal_reached_vid[None, :, :, :, :]
        if self.full_eval_mode:
            rewards = np.mean(rewards, axis=0)
            per_cell_rewards = {f"{cell}": rew for cell, rew in enumerate(rewards)}
            goal_reached = np.mean(goal_reached, axis=0)
            per_cell_goal_reached = {f"{cell}": reached for cell, reached in enumerate(goal_reached)}
            evaluation_metrics["evaluation"]["per_cell_rewards"] = per_cell_rewards
            evaluation_metrics["evaluation"]["per_cell_goal_reached"] = per_cell_goal_reached
            shift = 0
            scale = 1
            goal_reached_img = self._generate_highlighted_img(deepcopy(goal_reached_img), per_cell_goal_reached, shift, scale)
            evaluation_metrics["evaluation"]["per_cell_goal_reached_img"] = goal_reached_img

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
        episode.media["img"] = []
        episode.user_data["env_num_steps"] = 0
        needs_initialization = self.num_cells < 0
        
        if needs_initialization:
            # Initializes stuff needed for logging, that needs to know the arena's layout.
            arena = env.get_task()._maze_arena
            self.grid_h, self.grid_w = arena._maze.entity_layer.shape
            self.grid_positions = flatten_dict_of_lists(arena.find_token_grid_positions(RESPAWNABLE_TOKENS))
            self.world_positions = arena.grid_to_world_positions(self.grid_positions)
            self.num_cells = len(self.world_positions)
            self.eval_rewards = [0 for _ in range(self.num_cells)]
            self.goal_reached = [0 for _ in range(self.num_cells)]

        run_active_rl = np.random.random() < self.run_active_rl
        if self.is_evaluating and self.full_eval_mode:
            # Resets environment to all states, one by one.
            self.cell_index += 1
            initial_world_position = self.world_positions[self.cell_index % self.num_cells]
            initial_state = env.observation_space.sample()
            initial_state = env.combine_resettable_part(initial_state, initial_world_position)
            env.reset(initial_state=initial_state)
        elif not self.is_evaluating and run_active_rl:
            # Actually run Active RL and reset the environment
            new_states = self.reset_env(policy, env, episode)
            if ACTIVE_STATE_VISITATION_KEY not in episode.custom_metrics:
                episode.hist_data[ACTIVE_STATE_VISITATION_KEY] = []
            to_log, _ = env.separate_resettable_part(new_states)
            to_log = np.array(to_log)
            to_log = to_log[0] * self.grid_w + to_log[1]
            episode.hist_data[ACTIVE_STATE_VISITATION_KEY].append(to_log)

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
        episode.custom_metrics["reached_goal"] = float(env.reached_goal_last_ep)
        if self.is_evaluating:
            if self.full_eval_mode:
                self.eval_rewards[self.cell_index % self.num_cells] += episode.total_reward / (self.config["evaluation_duration"] // self.num_cells)
                self.goal_reached[self.cell_index % self.num_cells] = float(env.reached_goal_last_ep)
            pix = env.render()
            pix = np.transpose(pix, [2, 0, 1])
            episode.media["img"] = np.stack(episode.media["img"])

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
        env = base_env.get_sub_environments()[0]
        if episode.user_data["env_num_steps"] % 20 == 0:
            pix = env.render()
            pix = np.transpose(pix, [2, 0, 1])
            episode.media["img"].append(pix)
        episode.user_data["env_num_steps"] += 1

class SynergymCallback(ActiveRLCallback):
    def __init__(self, num_descent_steps: int=10, batch_size: int=64, no_coop: bool=False, planning_model=None, config={}, run_active_rl=False, planning_uncertainty_weight=1, device="cpu", args={}):
        super().__init__(num_descent_steps, batch_size, no_coop, planning_model, config, run_active_rl, planning_uncertainty_weight, device, args)

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
        episode.user_data["power"] = []
        episode.user_data["term_comfort"] = []
        episode.user_data["term_energy"] = []
        episode.user_data["num_comfort_violations"] = 0

        super().on_episode_start(
            worker = worker, base_env = base_env,
            policies = policies, episode=episode,
            **kwargs,
        )

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
        env = base_env.get_sub_environments()[0]
        info = episode.last_info_for()
        episode.user_data["power"].append(info["total_power"])
        episode.user_data["term_comfort"].append(info["comfort_penalty"])
        episode.user_data["term_energy"].append(info["total_power_no_units"])
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
        episode.custom_metrics["cum_power"] = np.sum(episode.user_data["power"])
        episode.custom_metrics["mean_power"] = np.mean(episode.user_data["power"])
        episode.custom_metrics["cum_comfort_penalty"] = np.sum(episode.user_data["term_comfort"])
        episode.custom_metrics["mean_comfort_penalty"] = np.mean(episode.user_data["term_comfort"])
        episode.custom_metrics["cum_power_penalty"] = np.sum(episode.user_data["term_energy"])
        episode.custom_metrics["mean_power_penalty"] = np.mean(episode.user_data["term_energy"])
        episode.custom_metrics["num_comfort_violations"] = episode.user_data["num_comfort_violations"]
        try:
            episode.custom_metrics['comfort_violation_time(%)'] = episode.user_data["num_comfort_violations"] / \
                episode.length * 100
        except ZeroDivisionError:
            episode.custom_metrics['comfort_violation_time(%)'] = np.nan
